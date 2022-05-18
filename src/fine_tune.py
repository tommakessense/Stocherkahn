from os import path
from time import time

import matplotlib.pyplot as plt
import tensorflow as tf

from utils import imp_args


@imp_args
def get_args(parser):
    parser.add_argument("datadir")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--img-width", type=int, default=150)
    parser.add_argument("--img-height", type=int, default=200)


def show_image_and_label(train_dataset):
    plt.figure(figsize=(10, 10))
    for images, labels in train_dataset.take(1):
        for i in range(6):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
        plt.show()


def get_data_augmentation():
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomContrast(0.3),
        # requires tf-nightly
        # tf.keras.layers.RandomBrightness(0.4),
    ])
    return data_augmentation

def create_additional_datasets(data_augmentation, train_datasets, size=8):
    augmented_images = []
    augmented_labels = []
    for image, label in train_datasets:
        for i in range(size):
            augmented_image = data_augmentation(image)
            augmented_images.append(augmented_image)
            augmented_labels.append(label)
    augmented_images = tf.stack(augmented_images)
    augmented_labels = tf.stack(augmented_labels)
    dataset = tf.data.Dataset.from_tensor_slices((augmented_images, augmented_labels))
    return dataset


def get_kfolded_datasets(train_data: list, train_targets: list,
                         num_val_samples: int, i: int, batch_size: int = 8):
    def batch(dataset, size):
        batch_set = []
        batched = []
        for img in dataset:
            if len(batch_set) == size:
                batched.append(batch_set)
                batch_set = []
            batch_set.append(img)
        return batched

    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = train_data[:i * num_val_samples]
    partial_train_data.extend(train_data[(i + 1) * num_val_samples:])
    partial_train_targets = train_targets[:i * num_val_samples]
    partial_train_targets.extend(train_targets[(i + 1) * num_val_samples:])
    return batch(val_data, batch_size), \
           batch(val_targets, batch_size), \
           batch(partial_train_data, batch_size), \
           batch(partial_train_targets, batch_size)


def create_datasets(image_size, train_dir, test_dir):
    tf.random.set_seed(round(time()))
    training = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                                 shuffle=True,
                                                                 batch_size=None,
                                                                 label_mode='binary',
                                                                 validation_split=0.25,
                                                                 seed=2,
                                                                 subset='training',
                                                                 image_size=image_size)
    validation = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                               shuffle=True,
                                                               batch_size=None,
                                                               label_mode='binary',
                                                               validation_split=0.25,
                                                               seed=2,
                                                               subset='validation',
                                                               image_size=image_size)
    testing = tf.keras.utils.image_dataset_from_directory(test_dir,
                                                                shuffle=True,
                                                                batch_size=None,
                                                                label_mode='binary',
                                                                image_size=image_size)
    return training, validation, testing


if __name__ == '__main__':

    args = get_args()
    IMG_SIZE = (args.img_width, args.img_height)
    BATCH_SIZE = args.batch_size

    train_datasets, val_datasets, test_datasets = create_datasets(
        IMG_SIZE,
        path.join(args.datadir, 'labeled'),
        path.join(args.datadir, 'tests')
    )
    class_names = train_datasets.class_names

    data_augmentation = get_data_augmentation()
    additional_datasets = create_additional_datasets(data_augmentation, train_datasets, 8)
    train_datasets = train_datasets.concatenate(additional_datasets)

    AUTOTUNE = tf.data.AUTOTUNE
    train_datasets = train_datasets.prefetch(buffer_size=AUTOTUNE)
    val_datasets = val_datasets.prefetch(buffer_size=AUTOTUNE)
    train_datasets = train_datasets.batch(BATCH_SIZE)
    val_datasets = val_datasets.batch(BATCH_SIZE)

    preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
    IMG_SHAPE = IMG_SIZE + (3,)
    base_model = tf.keras.applications.MobileNetV3Large(input_shape=IMG_SHAPE,
                                                        include_top=False,
                                                        weights='imagenet')
    image_batch, label_batch = next(iter(train_datasets))
    feature_batch = base_model(image_batch)
    base_model.trainable = False
    # base_model.summary()

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(1)

    inputs = tf.keras.Input(shape=IMG_SHAPE)
    # if args.augment:
    #     data_augmentation = get_data_augmentation()
    #     x = data_augmentation(inputs)
    # else:
    x = inputs
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    base_learning_rate = 0.0001
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy'])
    history = model.fit(train_datasets,
                        epochs=args.epochs,
                        validation_data=val_datasets)

    # Retrieve a batch of images from the test set
    test_datasets = test_datasets.batch(BATCH_SIZE)
    test_image_batch, test_label_batch = next(iter(test_datasets))
    predictions = model.predict_on_batch(test_image_batch).flatten()

    # Apply a sigmoid since our model returns logits
    predictions = tf.nn.sigmoid(predictions)
    predictions = tf.where(predictions < 0.5, 0, 1)

    predictions = predictions.numpy()
    correct_labels = test_label_batch.numpy()

    print(f'Predictions:\n{predictions}')
    print(f'Labels:\n[{" ".join([str(int(val[0])) for val in correct_labels])}]\n')

    plt.figure(figsize=(10, 10))
    plt.subplots_adjust(hspace=0.5)
    for i in range(BATCH_SIZE):
        plt.subplot(4, 4, i + 1)
        shaped_img = test_image_batch[i]
        plt.imshow(shaped_img.numpy().astype("uint8"))
        plt.title(class_names[predictions[i]])
        plt.axis("off")

    _ = plt.suptitle("Model predictions")
    plt.show()
