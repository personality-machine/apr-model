import tensorflow as tf
import tensorflow_probability as tfp

from utils import compose

PARAMS = {
    "base_learning_rate": 0.0001,
    "batch_size": 128,
    "img_height": 224,
    "img_width": 398,
}

def model():
    img_shape = (PARAMS["img_height"], PARAMS["img_height"], 3)

    feature_extractor = tf.keras.applications.MobileNetV2(
        input_shape=img_shape,
        include_top=False,
        weights='imagenet',
    )


    top_layers = tf.keras.models.Sequential([
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(240, activation='relu'),
        tf.keras.layers.Reshape((6,40)),
        tf.keras.layers.Softmax(axis=-1)
    ])

    # Declare model

    inputs = tf.keras.Input(shape=img_shape)
    x = feature_extractor(inputs, training=True)
    outputs = top_layers(x)

    # Fine-tune from this layer onwards
    fine_tune_at = 8

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in feature_extractor.layers[:fine_tune_at]:
        layer.trainable = False

    return tf.keras.Model(inputs, outputs)

def getLoss():
    return lambda y_true, y_pred: tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true,y_pred),axis=-1)
    # return tf.keras.losses.CategoricalCrossentropy()

def MAE_binned(y_true, y_pred, buckets=40):
    dist = tfp.distributions.Normal(loc=0.5,scale=0.15)
    true = dist.quantile(tf.cast(tf.math.argmax(y_true, axis=-1)/40,'float32'))
    pred = dist.quantile(tf.cast(tf.math.argmax(y_pred, axis=-1)/40,'float32'))
    result = tf.reshape(tf.math.abs(true - pred),[-1])
    #print(y_true,y_pred,result)
    return result
    #abs_difference = tf.math.abs(true - pred)
    #return [tf.reduce_mean(abs_difference)]

def compile(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(PARAMS["base_learning_rate"]),
        loss=getLoss(),
        metrics=[MAE_binned],
    )

def preprocess_image(image):
    return compose([
        tf.keras.layers.RandomCrop(PARAMS["img_height"], PARAMS["img_height"]),
        tf.keras.applications.mobilenet_v2.preprocess_input
    ])(image)

def bucket(x,buckets=40):
    dist = tfp.distributions.Normal(loc=0.5,scale=0.15)
    return tf.one_hot(tf.cast(dist.cdf(tf.cast(x,dtype='float32'))*buckets,dtype='int32'),buckets)

def preprocess_ds(image, label):
    return preprocess_image(image), bucket(label)