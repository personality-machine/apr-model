import tensorflow as tf
from utils import compose

PARAMS = {
    "base_learning_rate": 0.0001,
    "batch_size": 128,
    "img_height": 224,
    "img_width": 398,
}

def model():
    img_shape = (PARAMS["img_height"], PARAMS["img_height"], 3)

    feature_extractor = tf.keras.applications.EfficientNetB4(
        input_shape=img_shape,
        include_top=False,
        weights='imagenet',
    )


    top_layers = tf.keras.models.Sequential([
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(6, activation='sigmoid'),
    ])

    # Declare model

    inputs = tf.keras.Input(shape=img_shape)
    x = feature_extractor(inputs, training=True)
    outputs = top_layers(x)

    # Fine-tune from this layer onwards
    fine_tune_at = 80

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in feature_extractor.layers[:fine_tune_at]:
        layer.trainable = False

    return tf.keras.Model(inputs, outputs)

def compile(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(PARAMS["base_learning_rate"]),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()],
    )

def preprocess_image(image):
    return compose([
        tf.keras.layers.RandomCrop(PARAMS["img_height"], PARAMS["img_height"]),
        tf.keras.applications.resnet.preprocess_input
    ])(image)