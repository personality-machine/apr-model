import datetime
import importlib
import click
from absl import logging
import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path
import shutil

def load_data(params, data_dir, preprocess_ds):
    (ds_train, ds_val), ds_info = tfds.load(
        'first_impressions',
        split=['train', 'val'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
        data_dir=data_dir,
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

@click.group()
def cli():
    pass

@cli.command()
@click.option("--experiment", help="Experiment to run", required=True, type=str)
@click.option("--data_dir", help="Path to tfds dir", required=True, type=click.Path())
@click.option("--ckpt_base", help="Path to base checkpoint / log saving directory (.../experiment)", required=True, type=click.Path())
@click.option("--num_epochs", help="Number of epochs to train for", required=True, type=int)
@click.option("--save_every", help="Number of epochs between saves", required=True, type=float)
def train(
    experiment,
    data_dir,
    ckpt_base,
    num_epochs,
    save_every,
):
    """
    Train the model with specified options and hyperparameters
    """
    exp = importlib.import_module(f"experiments.{experiment}.experiment")
    ds_train, ds_val = load_data(
        exp.PARAMS, 
        data_dir, 
        lambda image, label: (exp.preprocess_image(image), label)
    )
    
    exp_base = Path(ckpt_base) / experiment
    if (Path(ckpt_base) / experiment).is_dir():
        click.confirm(
            "Existing directory found:\n"
            + f"- {exp_base}\n"
            + "Delete?",
            abort=True,
        )
        shutil.rmtree(exp_base)

    # Create the model
    model = exp.model()
    exp.compile(model)

    # Create a callback that saves the model's weights
    checkpoint_path = (exp_base / "checkpoints")
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=str(checkpoint_path) + "/cp-{epoch:04d}.ckpt",
                                                    save_weights_only=True,
                                                    verbose=1,
                                                    save_freq=int(save_every * len(ds_train)))
    log_dir = str(exp_base / "logs") + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Train the model with the new callback
    model.fit(
        ds_train,
        epochs=num_epochs,
        validation_data=ds_val,
        callbacks=[cp_callback, tensorboard_callback]
    )

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    cli()
