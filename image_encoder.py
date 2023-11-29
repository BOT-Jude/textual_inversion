import tensorflow as tf
import tensorflow_datasets as tfds

import pandas as pd
import matplotlib.pyplot as plt
import inspect
from tqdm import tqdm
import torch
from torch import nn
import os


class NullEncoder:
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim

    def __call__(self, data_root):
        image_paths = [os.path.join(data_root, file_path) for file_path in os.listdir(data_root)]
        return torch.ones(len(image_paths), self.embedding_dim)


def normalize_img(image, img_size):
    # Resize image to the desired img_size and normalize it
    # One hot encode the label
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.
    return image


class XceptionImageEncoder:
    def __init__(self, embedding_dim):
        assert embedding_dim == 2048
        model = {m[0]: m[1] for m in inspect.getmembers(tf.keras.applications, inspect.isfunction)}["Xception"]
        self.pre_trained_model = model(include_top=False, pooling='avg', input_shape=(128, 128, 3))
        self.pre_trained_model.trainable = False

    def __call__(self, data_root):

        image_paths = [os.path.join(data_root, file_path) for file_path in os.listdir(data_root)]

        embeddings = torch.zeros(len(image_paths), 2048)
        for i, path in enumerate(image_paths):
            image = tf.keras.utils.load_img(path)
            image = normalize_img(image, tf.constant([128, 128]))
            image = tf.expand_dims(image, 0)

            embeddings[i] = torch.from_numpy(self.pre_trained_model(image).numpy()[0])

        return embeddings


def is_image(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))


class LookupEncoder:
    def __init__(self, embedding_dim):
        assert embedding_dim == 50

    def __call__(self, data_root):

        # separate the data into images and characteristic data
        paths = [os.path.join(data_root, file_path) for file_path in os.listdir(data_root)]
        image_paths = self.image_paths = list(filter(is_image, paths))
        txt_files = list(filter(lambda x: x.endswith(".txt"), paths))
        assert len(txt_files) == 1, \
            "There must only be one txt file in the data directory"
        characteristic_file = txt_files[0]

        # read data from characteristic_file into dictionary
        self.photo_characteristics = {}

        with open(characteristic_file, "r") as data:
            data.readline()
            data.readline()
            for line in data:
                line = line.split()
                img_name = line.pop(0)
                descriptors = torch.tensor([int(c) for c in line], dtype=torch.float)

                # extract and normalize landmark coordinates
                landmarks = descriptors[-10:]
                landmarks[[0, 2, 4, 6, 8]] = (landmarks[[0, 2, 4, 6, 8]] / 178.0) - 0.5
                landmarks[[1, 3, 5, 7, 9]] = (landmarks[[1, 3, 5, 7, 9]] / 218.0) - 0.5

                # extract and map descriptors (-1, 1) -> (0, 1)
                attributes = descriptors[:-10]
                attributes = (attributes + 1.0) / 2.0

                self.photo_characteristics[img_name] = (attributes, landmarks)

        # select corresponding image data for embeddings matrix
        embeddings = torch.zeros(len(image_paths), 50)

        for i, path in enumerate(image_paths):
            img_name = path.split("/")[-1]
            attributes, landmarks = self.photo_characteristics[img_name]
            embeddings[i] = torch.cat([attributes, landmarks])

        return embeddings
