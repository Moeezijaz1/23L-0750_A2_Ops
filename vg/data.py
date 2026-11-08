"""Task 2.1 - tf.data input pipeline."""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from vg.config import BATCH, DATA_DIR, IMG_SIZE, SEED, SPLIT_DIR
from vg.preprocess import decode_resize

AUTOTUNE = tf.data.AUTOTUNE


def augmenter():
    # Mild, label-preserving transforms only. Blur/JPEG-style augmentation is avoided on purpose:
    # it can erase the very artefacts that separate AI images from real photos.
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.05),
        keras.layers.RandomZoom(0.1),
        keras.layers.RandomContrast(0.1),
    ], name="augment")


def load_split(name):
    df = pd.read_csv(SPLIT_DIR / f"{name}.csv")
    return [str(DATA_DIR / p) for p in df.path], df.label.values.astype("float32")


def make_ds(name, training=False, batch=BATCH, size=IMG_SIZE, augment=True):
    paths, labels = load_split(name)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (decode_resize(tf.io.read_file(p), size), y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch)
    if training and augment:  # augmentation on the TRAIN split only
        aug = augmenter()
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def labels_of(name):
    return np.array(load_split(name)[1])
