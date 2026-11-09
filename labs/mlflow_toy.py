"""Task 3.1 / 3.2 toy MLflow lab. Local:  python labs/mlflow_toy.py
Against DagsHub: set MLFLOW_TRACKING_URI / USERNAME / PASSWORD in .env first, then run the same command."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import mlflow  # noqa: E402
import mlflow.keras  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from tensorflow import keras  # noqa: E402

load_dotenv()
mlflow.set_experiment("toy-mnist")

(x, y), _ = keras.datasets.mnist.load_data()
# Explicitly cast to float32 and int32 to fix Keras 3 type promotion errors
x = (x[:10000].reshape(-1, 784) / 255.0).astype(np.float32)
y = y[:10000].astype(np.int32)

for lr in (1e-2, 1e-4):
    with mlflow.start_run(run_name=f"mnist_lr{lr}"):
        mlflow.set_tag("purpose", "toy-lab")
        mlflow.log_params({"lr": lr, "units": 64, "epochs": 5})
        m = keras.Sequential([
            keras.layers.Input((784,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(10, activation="softmax")
        ])
        m.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        h = m.fit(x, y, epochs=5, validation_split=0.2, verbose=0)
        for e in range(5):
            mlflow.log_metrics({"loss": float(h.history["loss"][e]), "val_accuracy": float(h.history["val_accuracy"][e])}, step=e)
        plt.figure()
        plt.plot(h.history["loss"], label="loss")
        plt.plot(h.history["val_loss"], label="val_loss")
        plt.legend()
        plt.savefig("toy_curve.png")
        mlflow.log_artifact("toy_curve.png")
        plt.close()
        mlflow.keras.log_model(m, "model")