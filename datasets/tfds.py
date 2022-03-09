import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path

def load_data(params, data_dir, preprocess_ds):
    (ds_train, ds_val), ds_info = tfds.load(
        'first_impressions',
        split=['train', 'val'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
        data_dir=Path(data_dir) / 'tfds',
    )

    ds_train = ds_train.map(
        preprocess_ds, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(params["batch_size"])
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_val = ds_val.map(
        preprocess_ds, num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = ds_val.batch(params["batch_size"])
    ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

    return (ds_train, ds_val)