import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd
import skvideo
import tensorflow as tf

skvideo.setFFmpegPath(str(Path(os.path.dirname(os.path.realpath(__file__))) / 'first_impressions/libs/ffmpeg-5.0-amd64-static'))
import skvideo.io

import datasets.tfds

LABELS = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism', 'interview']
IMG_SIZE = [224,398]

def load_from_idx(idx, split, df, first_impressions_dir):
    row = df.iloc[idx]
    return (
        skvideo.io.vread(str(first_impressions_dir / split / row.name)),
        np.array([row[i] for i in LABELS])
    )

def ds_from_video(video, labels):
    return tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(video), 
        tf.data.Dataset.from_tensors(labels).repeat()
    ))

def _fixup_shape(x, y):
    x.set_shape([None, None, 3])
    y.set_shape([6])
    return x, y

def ds_from_idx_batch(batch, split, df, first_impressions_dir):
    ds = tf.data.Dataset.from_tensor_slices(batch)
    return ds.map(
        lambda i: tf.numpy_function(
            func=lambda x: load_from_idx(x, split, df, first_impressions_dir), 
            inp=[i], 
            Tout=[
                tf.uint8,
                tf.float64,
            ]
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    ).flat_map(#.interleave(
        ds_from_video,
        #num_parallel_calls=tf.data.AUTOTUNE,
        #deterministic=False
    ).map(
        _fixup_shape,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    ).map(
        lambda img, label: (tf.image.resize(img, IMG_SIZE), label),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    ).shuffle(tf.cast(len(batch), tf.int64) * 450)

def gen_ds(first_impressions_dir, split, df, params):
    z = list(range(len(df))) # the index generator
    ds = tf.data.Dataset.from_generator(lambda: z, tf.uint8)
    return ds.batch(params["load_buffer"]).flat_map(
        lambda x: ds_from_idx_batch(x, split, df, first_impressions_dir)
    )

def load_data(params, data_dir, preprocess_ds):
    # shuffle first `shuffle_buffer` frames
    # load `interleave_frames` frames from each video

    first_impressions_dir = Path(data_dir) / 'first-impressions'

    train_df = pd.read_csv(first_impressions_dir / 'train/labels.csv', index_col=0)
    val_df = pd.read_csv(first_impressions_dir / 'val/labels.csv', index_col=0)

    ds_train = gen_ds(first_impressions_dir, "train", train_df, params)
    ds_val = gen_ds(first_impressions_dir, "val", val_df, params)

    ds_train = ds_train.map(
        preprocess_ds, num_parallel_calls=tf.data.AUTOTUNE)
    # ds_train = ds_train.shuffle(params["load_buffer"] * 450)
    ds_train = ds_train.batch(params["batch_size"])
    ds_train = ds_train.cache()
    ds_train = ds_train.repeat()
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_val = ds_val.map(
        preprocess_ds, num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = ds_val.batch(params["batch_size"])
    ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

    tfds_train, tfds_val = datasets.tfds.load_data(params, data_dir, preprocess_ds)

    return (ds_train, tfds_val)
