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


class CelebALookupEncoder:
    def __init__(self, embedding_dim):
        assert embedding_dim == 50

    def __call__(self, data_root):

        # find image paths and landmarks.txt and attributes.txt
        paths = [os.path.join(data_root, file_path) for file_path in os.listdir(data_root)]
        image_paths = self.image_paths = list(filter(is_image, paths))
        landmarks_file = list(filter(lambda x: x.endswith("list_landmarks_align_celeba.txt"), paths))[0]
        attributes_file = list(filter(lambda x: x.endswith("list_attr_celeba.txt"), paths))[0]

        # create dictionary for landmark data
        self.photo_landmarks = {}
        with open(landmarks_file, "r") as f:
            f.readline()
            f.readline()
            for line in f:
                line = line.split()
                img_name = line.pop(0)

                # extract and normalize landmark coordinates
                landmarks = torch.tensor([int(c) for c in line], dtype=torch.float)
                landmarks[[0, 2, 4, 6, 8]] = (landmarks[[0, 2, 4, 6, 8]] / 178.0) - 0.5
                landmarks[[1, 3, 5, 7, 9]] = (landmarks[[1, 3, 5, 7, 9]] / 218.0) - 0.5

                self.photo_landmarks[img_name] = landmarks

        # create dictionary for attribute data
        self.photo_attributes = {}
        with open(attributes_file, "r") as f:
            f.readline()
            f.readline()
            for line in f:
                line = line.split()
                img_name = line.pop(0)

                # extract and map attributes (-1, 1) -> (0, 1)
                attributes = torch.tensor([int(c) for c in line], dtype=torch.float)
                attributes = (attributes + 1.0) / 2.0

                self.photo_attributes[img_name] = attributes

        # select corresponding image data for embeddings matrix
        embeddings = torch.zeros(len(image_paths), 50)

        for i, path in enumerate(image_paths):
            img_name = path.split("/")[-1]
            landmarks = self.photo_landmarks[img_name]
            attributes = self.photo_attributes[img_name]
            embeddings[i] = torch.cat([attributes, landmarks])

        return embeddings
