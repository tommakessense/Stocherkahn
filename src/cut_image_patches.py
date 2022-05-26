from os import path

import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

image_id = "VID_20220514_140653__1__300"
directory = "/home/tom/Data/Stocherkahn"
height = 135
width = 240
patch_directory = path.join(directory, "patches")
image = tf.keras.utils.load_img(f"{directory}/{image_id}.jpg")
image = tf.expand_dims(image, axis=0)
patches = tf.image.extract_patches(images=image,
                         sizes=[1, height, width, 1],
                         strides=[1, height, width, 1],
                         rates=[1, 1, 1, 1],
                         padding='SAME')

# plt.figure(figsize=(10, 10))
total = 0
for patch in patches:
    count = 0
    batch_size = patch.shape.as_list()[0]
    for r in range(batch_size):
        for c in range(batch_size):
            plt.figure(figsize=(height//100, width//100))
            # ax = plt.subplot(batch_size, batch_size, count+1)
            img_arr = tf.reshape(patch[r, c], shape=(height, width, 3)).numpy().astype("uint8")
            # plt.imshow(tf.reshape(patch[r, c], shape=(height, width, 3)).numpy().astype("uint8"))
            plt.imsave(path.join(patch_directory, f"{image_id}_patch{total}.png"), img_arr)
            count += 1
            total += 1

# plt.show()

# for idx, patch in enumerate(next(iter(patches))):
#     image = patch.numpy().astype("uint8")
#     cv2.imwrite(f'/home/tom/Data/Stocherkahn/patches/image_{idx}.png', image)
