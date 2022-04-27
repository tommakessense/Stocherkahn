import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.ops.gen_dataset_ops import BatchDataset

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
    for image_batch, label in train_datasets.take(1):
        for image in image_batch:
            for i in range(size):
                augmented_image = data_augmentation(image)
                augmented_images.append((augmented_image, label))
    batch = []
    dataset = []
    for image, label in augmented_images:
        if len(batch) == BATCH_SIZE:
            dataset.append(batch)
            batch = []
        batch.append((image, label))
    dataset.append(batch)
    return dataset


def get_kfolded_datasets(train_data: list, train_targets: list, num_val_samples: int, i: int):
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)
    return val_data, val_targets, partial_train_data, partial_train_targets


def convert_train_datasets(train_datasets):
    train_data = []
    train_targets = []
    for image, label in train_datasets:
        train_data.append(image)
        train_targets.append(label)
    return train_data, train_targets


if __name__ == '__main__':
    train_datasets = tf.keras.utils.image_dataset_from_directory(datadir,
                                                                shuffle=True,
                                                                batch_size=BATCH_SIZE,
                                                                image_size=IMG_SIZE)
    class_names = train_datasets.class_names

    # add more training data
    data_augmentation = get_data_augmentation()
    additional_datasets = create_additional_datasets(data_augmentation, train_datasets)
    train_datasets.append(additional_datasets)
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_datasets.prefetch(buffer_size=AUTOTUNE)
    #
    # preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
    # IMG_SHAPE = IMG_SIZE + (3,)
    # base_model = tf.keras.applications.MobileNetV3Large(input_shape=IMG_SHAPE,
    #                                                     include_top=False,
    #                                                     weights='imagenet')
    # base_model.trainable = False
    # base_model.summary()
    # train_data, train_targets = convert_train_datasets(train_datasets)
    # num_val_samples = len(train_datasets) // k
    #
    # for i in range(k):
    #     print(f"Processing fold #{i}")
    #     val_data, val_targets, partial_train_data, partial_train_targets = get_kfolded_datasets(train_data, train_targets, num_val_samples)
    #     # model compile
    #
    #     # model fit
    #
