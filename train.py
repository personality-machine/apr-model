import tensorflow as tf
import tensorflow_datasets as tfds

import datasets.first_impressions

import os
from pathlib import Path
import datetime

DATA_DIR = '/rds/user/elyro2/hpc-work/personality-machine/tfds'
CHECKPOINT_DIR = '/rds/user/elyro2/hpc-work/personality-machine/experiments/exp_02_19_baseline_resnet'
IMG_HEIGHT=224
IMG_WIDTH=398
NUM_EPOCHS=500
BATCH_SIZE=128
BASE_LEARNING_RATE=0.0001
SAVE_FREQ=BATCH_SIZE

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
    return tf.keras.applications.resnet.preprocess_input(
        tf.keras.layers.RandomCrop(IMG_HEIGHT, IMG_HEIGHT)(tf.cast(image, tf.float32) / 255., training=True)
    ), label

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

feature_extractor = tf.keras.applications.resnet50.ResNet50(
    input_shape=(IMG_HEIGHT, IMG_HEIGHT, 3),
    include_top=False,
    weights='imagenet',
)

top_layers = tf.keras.models.Sequential([
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(6, activation='sigmoid'),
])

# Model

inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_HEIGHT, 3))
x = feature_extractor(inputs, training=True)
outputs = top_layers(x)

# Fine-tune from this layer onwards
fine_tune_at = 80

# Freeze all the layers before the `fine_tune_at` layer
for layer in feature_extractor.layers[:fine_tune_at]:
  layer.trainable = False

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(BASE_LEARNING_RATE),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.MeanAbsoluteError()],
)

checkpoint_path = CHECKPOINT_DIR + "/cp-{epoch:04d}.ckpt"

print(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 save_freq=SAVE_FREQ)

log_dir = CHECKPOINT_DIR + "/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model with the new callback
model.fit(
    ds_train,
    epochs=NUM_EPOCHS,
    validation_data=ds_val,
    callbacks=[cp_callback, tensorboard_callback]
)