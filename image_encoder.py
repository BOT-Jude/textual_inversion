import tensorflow as tf
import tensorflow_datasets as tfds

import pandas as pd
import matplotlib.pyplot as plt
import inspect
from tqdm import tqdm
import torch
from torch import nn


def normalize_img(image, img_size):
    # Resize image to the desired img_size and normalize it
    # One hot encode the label
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.
    return image


class ImageEncoder:
    def __init__(self, embedding_dim):
        assert embedding_dim == 1000
        model = {m[0]: m[1] for m in inspect.getmembers(tf.keras.applications, inspect.isfunction)}["Xception"]
        self.pre_trained_model = model(include_top=False, pooling='avg', input_shape=(128, 128, 3))
        self.pre_trained_model.trainable = False

    def __call__(self, image_paths):
        embeddings = torch.zeros(len(image_paths), 1000)
        for i, path in enumerate(image_paths):
            image = tf.keras.utils.load_img(path)
            image = normalize_img(image, tf.constant([128, 128]))
            image = tf.expand_dims(image, 0)

            embeddings[i] = torch.from_numpy(self.pre_trained_model(image).numpy()[0])

        return embeddings


class NullEncoder:
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim

    def __call__(self, image_paths):
        return torch.ones(len(image_paths), self.embedding_dim)
