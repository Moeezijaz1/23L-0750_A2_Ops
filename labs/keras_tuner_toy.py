"""Task 2.7 toy: random search on Fashion-MNIST for learning rate and number of units."""
import keras_tuner as kt
from tensorflow import keras

(x, y), _ = keras.datasets.fashion_mnist.load_data()
x, y = x[:10000] / 255.0, y[:10000]


def build(hp):
    m = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)),
                          keras.layers.Dense(hp.Int("units", 32, 256, step=32), activation="relu"),
                          keras.layers.Dense(10, activation="softmax")])
    m.compile(keras.optimizers.Adam(hp.Float("lr", 1e-4, 1e-2, sampling="log")),
              "sparse_categorical_crossentropy", ["accuracy"])
    return m


t = kt.RandomSearch(build, "val_accuracy", max_trials=8, overwrite=True, directory="kt_toy", project_name="fmnist")
t.search(x, y, epochs=4, validation_split=0.2)
t.results_summary()
