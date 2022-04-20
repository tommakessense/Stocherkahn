from collections import OrderedDict
from typing import Union

import tensorflow as tf
import tensorflow_hub as hub

import requests
from PIL import Image
from io import BytesIO
import os, sys

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import Tensor

original_image_cache = {}


def preprocess_image(image):
    image = np.array(image)
    # reshape into shape [batch_size, height, width, num_channels]
    img_reshaped = tf.reshape(image, [1, image.shape[0], image.shape[1], image.shape[2]])
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    image = tf.image.convert_image_dtype(img_reshaped, tf.float32)
    return image


def load_image_from_url(img_url):
    """Returns an image with shape [1, height, width, num_channels]."""
    user_agent = {'User-agent': 'Colab Sample (https://tensorflow.org)'}
    response = requests.get(img_url, headers=user_agent)
    image = Image.open(BytesIO(response.content))
    image = preprocess_image(image)
    return image


def load_image(image_url, image_size=256, dynamic_size=False, max_dynamic_size=512):
    """Loads and preprocesses images."""
    # Cache image file locally.
    if image_url in original_image_cache:
        img = original_image_cache[image_url]
    elif image_url.startswith('https://'):
        img = load_image_from_url(image_url)
    else:
        fd = tf.io.gfile.GFile(image_url, 'rb')
        img = preprocess_image(Image.open(fd))
    original_image_cache[image_url] = img
    # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
    img_raw = img
    if tf.reduce_max(img) > 1.0:
        img = img / 255.
    if len(img.shape) == 3:
        img = tf.stack([img, img, img], axis=-1)
    if not dynamic_size:
        img = tf.image.resize_with_pad(img, image_size, image_size)
    elif img.shape[1] > max_dynamic_size or img.shape[2] > max_dynamic_size:
        img = tf.image.resize_with_pad(img, max_dynamic_size, max_dynamic_size)
    return img, img_raw


def show_image(image, title=''):
    image_size = image.shape[1]
    w = (image_size * 6) // 320
    plt.figure(figsize=(w, w))
    plt.imshow(image[0], aspect='equal')
    plt.axis('off')
    plt.title(title)
    plt.show()


# required image size from mobilenet_v3
image_size = 224
max_dynamic_size = 512
dynamic_size = False
model_handle = "/home/tom/IdeaProjects/mobilenet_v3"


def get_classes():
    labels_file = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
    downloaded_file = tf.keras.utils.get_file("labels.txt", origin=labels_file)
    with open(downloaded_file) as f:
        labels = f.readlines()
        classes = [l.strip() for l in labels]
    return classes


def get_full_paths():
    _full_paths = []
    base_dir = "/home/tom/Desktop/images/water/canoe"
    for _file in os.listdir(base_dir):
        img_url = f'{base_dir}/{_file}'
        _full_paths.append(img_url)
    return _full_paths


# show_image(image, 'Scaled image')


def run_model(image, classes, classifier):
    probabilities = tf.nn.softmax(classifier(image)).numpy()

    top_results = tf.argsort(probabilities, axis=-1, direction="DESCENDING")[0][:10].numpy()
    np.array(classes)

    # Some models include an additional 'background' class in the predictions, so
    # we must account for this when reading the class labels.
    includes_background_class = probabilities.shape[1] == 1001
    results = OrderedDict()
    for i, item in enumerate(top_results):
        class_index = item if includes_background_class else item + 1
        line = f'({i + 1}) {class_index:4} - {classes[class_index]}: {probabilities[0][top_results][i]}'
        # print(line)
        results[classes[class_index]] = probabilities[0][top_results][i]
    return results

def warmup_classifier(img_path, classifier):
    img, _ = load_image(img_path, image_size, dynamic_size, max_dynamic_size)
    input_shape = img.shape
    warmup_input = tf.random.uniform(input_shape, 0, 1.0)
    warmup_logits = classifier(warmup_input).numpy()


if __name__ == '__main__':
    classes = get_classes()
    full_paths = get_full_paths()
    print(len(full_paths))
    classifier = hub.load(model_handle)
    warmup_classifier(full_paths[0], classifier)
    class_count = {}
    count = 0
    for idx, img_path in enumerate(full_paths):
        print(f'{idx +1} - Processing image {img_path}')
        img, img_raw = load_image(img_path, image_size, dynamic_size, max_dynamic_size)
        detected_objects = run_model(img, classes, classifier)
        for object_name in detected_objects:
            class_count.setdefault(object_name, 0)
            class_count[object_name] += 1

    for cls, count in sorted(class_count.items(), key=lambda x: x[1], reverse=True):
        print(f"{cls}: {count}")
