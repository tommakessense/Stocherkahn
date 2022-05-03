import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.ops.gen_dataset_ops import BatchDataset
from tensorflow import keras
from keras import layers

datadir = '/home/tom/Desktop/images/water'
BATCH_SIZE = 8
IMG_SIZE = (150, 200)
k = 4
num_epochs = 100

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
        tf.keras.layers.RandomBrightness(0.4),
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

def build_model_kfold(base_model, shape, data_augmentation):
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(1)
    inputs = tf.keras.Input(shape=shape)
    # x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v3.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    kmodel = tf.keras.Model(inputs, outputs)
    kmodel.compile(optimizer="rmsprop", loss="mse", metrics=["accuracy"])
    return kmodel

def get_kfolded_datasets(train_data: list, train_targets: list, num_val_samples: int, i: int, batch_size: int = 8):

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
           batch(val_targets, batch_size),\
           batch(partial_train_data, batch_size), \
           batch(partial_train_targets, batch_size)


def convert_train_datasets(train_datasets):
    train_data = []
    train_targets = []
    for image, label in train_datasets:
        train_data.append(image)
        train_targets.append(label)
    return train_data, train_targets


if __name__ == '__main__':
    train_datasets = tf.keras.utils.image_dataset_from_directory(datadir,
                                                                shuffle=False,
                                                                batch_size=None,
                                                                 validation_split=0.25,
                                                                 subset='training',
                                                                image_size=IMG_SIZE)
    val_datasets = tf.keras.utils.image_dataset_from_directory(datadir,
                                                                 shuffle=False,
                                                                 batch_size=None,
                                                                 validation_split=0.25,
                                                                 subset='validation',
                                                                 image_size=IMG_SIZE)
    class_names = train_datasets.class_names

    # add more training data
    data_augmentation = get_data_augmentation()
    additional_datasets = create_additional_datasets(data_augmentation, train_datasets, 16)
    train_datasets = train_datasets.concatenate(additional_datasets)
    AUTOTUNE = tf.data.AUTOTUNE
    train_datasets = train_datasets.prefetch(buffer_size=AUTOTUNE)
    val_datasets = val_datasets.prefetch(buffer_size=AUTOTUNE)
    train_datasets = train_datasets.batch(BATCH_SIZE)
    val_datasets = val_datasets.batch(BATCH_SIZE)
    #train_data, train_targets = convert_train_datasets(train_datasets)

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
    x = tf.keras.applications.mobilenet_v3.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
    initial_epochs = 15
    # split = len(train_dataset) // 4
    # validation_split = train_dataset.range(0, split)
    # train_split = train_dataset.range(split, len(train_dataset) - 1)
    history = model.fit(train_datasets,
                        epochs=initial_epochs,
                        validation_data=val_datasets)

    # num_val_samples = len(train_datasets) // k
    # all_mae_histories = []
    # for i in range(k):
    # print(f"Processing fold #{i}")
    # val_data, val_targets, partial_train_data, partial_train_targets = get_kfolded_datasets(train_data, train_targets, num_val_samples, i, BATCH_SIZE)
    # model = build_model_kfold(base_model, IMG_SHAPE, data_augmentation)
    # history = model.fit(partial_train_data, partial_train_targets,
    #                 validation_data=(val_data, val_targets),
    #                 epochs=num_epochs, batch_size=BATCH_SIZE, verbose=0)
    # accuracy_history = history.history["val_accuracy"]
    # all_mae_histories.append(accuracy_history)
        # model compile

