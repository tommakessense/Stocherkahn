import tensorflow as tf
import cv2

image = tf.keras.utils.load_img("/home/tom/Data/Stocherkahn/VID_20220514_140653__1__300.jpg")
image = tf.expand_dims(image, axis=0)
patches = tf.image.extract_patches(images=image,
                         sizes=[1, 135, 240, 1],
                         strides=[1, 67, 120, 1],
                         rates=[1, 1, 1, 1],
                         padding='VALID')
for idx, patch in enumerate(next(iter(patches))):
    image = patch.numpy().astype("uint8")
    cv2.imwrite(f'/home/tom/Data/Stocherkahn/patches/image_{idx}.png', image)
