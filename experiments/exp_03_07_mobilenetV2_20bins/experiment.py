import tensorflow as tf
import tensorflow_probability as tfp

from utils import compose

PARAMS = {
    "base_learning_rate": 0.001,
    "batch_size": 1024,
    "img_height": 224,
    "img_width": 398,
}

NO_BUCKETS = 20
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
        tf.keras.layers.Dense(6*NO_BUCKETS, activation='relu'),
        tf.keras.layers.Reshape((6,NO_BUCKETS)),
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

def compile(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(PARAMS["base_learning_rate"]),
        loss=getLoss(),
        metrics=[MAE_binned, tf.keras.metrics.CategoricalAccuracy()],
    )

def preprocess_image(image):
    return compose([
        tf.keras.layers.RandomCrop(PARAMS["img_height"], PARAMS["img_height"]),
        tf.keras.applications.mobilenet_v2.preprocess_input
    ])(image)

def MAE_binned(y_true, y_pred, buckets=NO_BUCKETS):
    # y_true, y_pred are [*, 6, buckets]
    dist = tfp.distributions.Normal(loc=0.5,scale=0.15)
    lbp, ubp = dist.cdf(0), dist.cdf(1)
    scaled_quantile = lambda x: (dist.cdf(x) - lbp) / (ubp - lbp)

    true, pred = [
        scaled_quantile((tf.cast(tf.math.argmax(y, axis=-1), 'float32') + 0.5) / buckets)
        for y in [y_true, y_pred]
    ]
    
    return tf.math.abs(true - pred)
    #print(y_true,y_pred,result)
    #abs_difference = tf.math.abs(true - pred)
    #return [tf.reduce_mean(abs_difference)]

def bucket(x,buckets=NO_BUCKETS):
    """
    Input
        x: float tf.Tensor[...dims] in [0,1]
    Returns: one-hot tf.Tensor[...dims, buckets]

    Buckets: 
        0: [0,1/buckets)
        1: [1/buckets, 2/buckets) 
        ... 
        buckets-1: [(buckets-1)/buckets, 1]
    """
    dist = tfp.distributions.Normal(loc=0.5,scale=0.15)

    lbp, ubp = dist.cdf(0), dist.cdf(1)
    scaled_cdf = lambda x: (dist.cdf(x) - lbp) / (ubp - lbp)

    return compose([
        lambda x: tf.cast(x,dtype='float32'),
        lambda x: scaled_cdf(x) * buckets,
        lambda x: tf.cast(x,dtype='int32'),
        lambda x: tf.minimum(x, buckets-1),
        lambda x: tf.one_hot(x,buckets)
    ])(x)

def preprocess_ds(image, label):
    return preprocess_image(image), bucket(label)