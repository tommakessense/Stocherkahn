import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

datadir = '/home/tom/Desktop/images/water'

BATCH_SIZE = 32
IMG_SIZE = (150, 200)

train_dataset = tf.keras.utils.image_dataset_from_directory(datadir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE)

class_names = train_dataset.class_names

# plt.figure(figsize=(10, 10))
# for images, labels in train_dataset.take(1):
#     for i in range(6):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_names[labels[i]])
#         plt.axis("off")
#     plt.show()

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomContrast(0.3),
    tf.keras.layers.RandomBrightness(0.4),
])

for image, _ in train_dataset.take(5):
    plt.figure(figsize=(10, 10))
    first_image = image[0]
    for i in range(6):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0] / 255)
        plt.axis('off')
    plt.show()
