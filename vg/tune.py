"""Task 2.7 (apply) - Keras Tuner random search over learning rate, dropout, #unfrozen layers (>= 8 trials).
Usage: python -m vg.tune --trials 10"""
import argparse

import keras_tuner as kt
from tensorflow import keras

from vg import models
from vg.config import IMG_SIZE, MODEL_DIR, SEED
from vg.data import make_ds


def build(hp):
    model, base = models.build_transfer("mobilenetv2", IMG_SIZE, dropout=hp.Float("dropout", 0.1, 0.6, step=0.1))
    n = hp.Choice("unfrozen_layers", [0, 10, 30, 60])
    if n == 0:
        base.trainable = False
    else:
        models.unfreeze_last_block(base, "mobilenetv2", n_layers=n)
    models.compile_model(model, hp.Float("lr", 1e-5, 1e-3, sampling="log"))
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=6)
    a = ap.parse_args()
    tuner = kt.RandomSearch(build, objective="val_loss", max_trials=a.trials, seed=SEED,
                            directory=str(MODEL_DIR / "tuner"), project_name="visionguard", overwrite=True)
    tuner.search(make_ds("train", True), validation_data=make_ds("val"), epochs=a.epochs,
                 callbacks=[keras.callbacks.EarlyStopping("val_loss", patience=2)])
    tuner.results_summary()
    print("BEST:", tuner.get_best_hyperparameters(1)[0].values)
