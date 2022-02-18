import tensorflow as tf
import tensorflow_datasets as tfds

from einops.layers.keras import Rearrange, Reduce

import datasets.first_impressions

import os
from pathlib import Path
import datetime

DATA_DIR = '/rds/user/elyro2/hpc-work/personality-machine/tfds'
CHECKPOINT_DIR = '/rds/user/elyro2/hpc-work/personality-machine/experiments/exp_02_18_baseline_resnet'
IMG_HEIGHT=224
IMG_WIDTH=398
NUM_EPOCHS=500
BATCH_SIZE=128
BASE_LEARNING_RATE=0.001
SAVE_FREQ=5*BATCH_SIZE

(ds_train, ds_val), ds_info = tfds.load(
    'first_impressions',
    split=['train', 'val'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
    data_dir=DATA_DIR,
)

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_val = ds_val.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_val = ds_val.batch(BATCH_SIZE)
ds_val = ds_val.cache()
ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

# Layers

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomCrop(IMG_HEIGHT, IMG_HEIGHT)
])

preprocess_input = tf.keras.applications.resnet.preprocess_input

feature_extractor = tf.keras.applications.resnet50.ResNet50(
    input_shape=(IMG_HEIGHT, IMG_HEIGHT, 3),
    include_top=False,
    weights='imagenet',
)

top_layers = tf.keras.models.Sequential([
    Reduce('b h w f -> b f', 'mean'),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(6, activation='sigmoid'),
])

# Model

inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = feature_extractor(x)
outputs = top_layers(x)

model = tf.keras.Model(inputs, outputs)

# Layers

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomCrop(IMG_HEIGHT, IMG_HEIGHT)
])

preprocess_input = tf.keras.applications.resnet.preprocess_input

feature_extractor = tf.keras.applications.resnet50.ResNet50(
    input_shape=(IMG_HEIGHT, IMG_HEIGHT, 3),
    include_top=False,
    weights='imagenet',
)

top_layers = tf.keras.models.Sequential([
    Reduce('b h w f -> b f', 'mean'),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(6, activation='sigmoid'),
])

# Model

inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = feature_extractor(x)
outputs = top_layers(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(BASE_LEARNING_RATE),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.MeanAbsoluteError()],
)

checkpoint_path = Path(CHECKPOINT_DIR) / "/cp-{epoch:04d}.ckpt"

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 save_freq=SAVE_FREQ)

log_dir = Path(CHECKPOINT_DIR) / "logs" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model with the new callback
model.fit(
    ds_train,
    epochs=NUM_EPOCHS,
    validation_data=ds_val,
    callbacks=[cp_callback, tensorboard_callback]
)