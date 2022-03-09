import datetime
import importlib
import click
from absl import logging
import tensorflow as tf
from pathlib import Path
import shutil
import tensorflowjs as tfjs
import datasets.tfds

import wandb
from wandb.keras import WandbCallback

@click.group()
def cli():
    pass

@cli.command()
@click.option("--experiment", help="Experiment to run", required=True, type=str)
@click.option("--data_dir", help="Path to personality-machine dir", required=True, type=click.Path())
@click.option("--ckpt_base", help="Path to base checkpoint / log saving directory (.../experiment)", required=True, type=click.Path())
@click.option("--num_epochs", help="Number of epochs to train for", required=True, type=int)
@click.option("--save_every", help="Number of epochs between saves", required=True, type=float)
@click.option("--enable_wandb/--disable_wandb", help="Whether or not to log results in WandB", default=True, type=bool)
def train(
    experiment,
    data_dir,
    ckpt_base,
    num_epochs,
    save_every,
    enable_wandb,
):
    """
    Train the model with specified options and hyperparameters
    """
    exp = importlib.import_module(f"experiments.{experiment}.experiment")
    if enable_wandb:
        wandb.init(project="firstimpressions", entity="personalitymachine", settings=wandb.Settings(start_method="fork"))
        wandb.run.name = experiment
        wandb.config = {
            **exp.PARAMS,
            "experiment": experiment,
            "data_dir": data_dir,
            "ckpt_base": ckpt_base,
            "num_epochs": num_epochs,
            "save_every": save_every,
        }

    if "load_data" in exp.PARAMS:
        load_data_fn = exp.PARAMS["load_data"]
    else:
        load_data_fn = datasets.tfds.load_data

    ds_train, ds_val = load_data_fn(
        exp.PARAMS, 
        data_dir, 
        exp.preprocess_ds if hasattr(exp, 'preprocess_ds') else lambda image, label: (exp.preprocess_image(image), label)
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
                                                    save_freq='epoch')
    log_dir = str(exp_base / "logs") + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Train the model with the new callback
    callbacks = [cp_callback, tensorboard_callback]
    if enable_wandb:
        callbacks.append(WandbCallback(
            save_model=True,
            monitor="val_loss",
            mode="min",
            # generator=ds_val,
            input_type="image"))

    model.fit(
        ds_train,
        epochs=500000,
        steps_per_epoch=100,
        validation_data=ds_val,
        callbacks=callbacks,
        use_multiprocessing=True,
        workers=10,
    )

@cli.command()
@click.option("--experiment", help="Experiment to run", required=True, type=str)
@click.option("--ckpt_base", help="Path to base checkpoint / log saving directory (.../experiment)", required=True, type=click.Path())
@click.option("--ckpt", help="Checkpoint to export", required=True, type=int)
@click.option("--zip/--no_zip", help="Create a zip archive", default=True)
@click.option("--saved_model_path", help="Saved model path", required=True, type=click.Path())
@click.option("--to_tfjs/--no_tfjs", help="Export to tfjs format", default=True)
def export(
    experiment,
    ckpt_base,
    ckpt,
    zip,
    saved_model_path,
    to_tfjs
):
    """
    Train the model with specified options and hyperparameters
    """
    exp = importlib.import_module(f"experiments.{experiment}.experiment")
    saved_model_path = Path(saved_model_path)
    js_path = saved_model_path.parent / (saved_model_path.name + "_js")

    zip_path = saved_model_path.parent / f"{saved_model_path.name}.zip"
    if saved_model_path.is_dir():
        click.confirm(
            "Existing directory found:\n"
            + f"- {saved_model_path}\n"
            + "Delete?",
            abort=True,
        )
        shutil.rmtree(saved_model_path)
        zip_path.unlink()

    # Create the model
    model = exp.model()
    exp.compile(model)

    # Save the model
    checkpoint_path = Path(ckpt_base) / experiment / "checkpoints"
    checkpoint_to_use = max(
        x for x in set(
            int(i.name.split(".")[0].split("-")[1]) for i in checkpoint_path.glob("cp-*.ckpt.index")
        ) if x <= ckpt)
    model.load_weights(checkpoint_path / f"cp-{checkpoint_to_use:04d}.ckpt")
    
    model.save(saved_model_path)
    if to_tfjs:
        tfjs.converters.save_keras_model(model, js_path)
    
    if zip:
        shutil.make_archive(saved_model_path, 'zip', saved_model_path)
        if to_tfjs:
            shutil.make_archive(js_path, 'zip', js_path)

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    cli()
