"""Task 3 - MLflow helpers. Credentials come ONLY from environment variables (.env, git-ignored)."""
import subprocess
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from tensorflow import keras

from vg.config import ROOT


def setup(experiment=None):
    load_dotenv(ROOT / ".env")
    import os
    if not os.getenv("MLFLOW_TRACKING_URI"):
        print("MLFLOW_TRACKING_URI not set -> logging to local ./mlruns")
    mlflow.set_experiment(experiment or os.getenv("MLFLOW_EXPERIMENT_NAME", "VisionGuard"))


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def dvc_data_version():
    """md5 recorded in data/raw.dvc = the exact dataset version used."""
    f = ROOT / "data" / "raw.dvc"
    if f.exists():
        for line in f.read_text().splitlines():
            if "md5:" in line:
                return line.split("md5:")[1].strip()
    return "untracked"


class EpochLogger(keras.callbacks.Callback):
    """Manual per-epoch logging (3.3a). `offset` keeps the step axis continuous across training stages."""

    def __init__(self, offset=0):
        super().__init__()
        self.offset = offset

    def on_epoch_end(self, epoch, logs=None):
        logs = dict(logs or {})
        try:
            logs["lr"] = float(self.model.optimizer.learning_rate.numpy())
        except Exception:  # noqa: BLE001
            pass
        mlflow.log_metrics({k: float(v) for k, v in logs.items()}, step=self.offset + epoch)


def log_pip_freeze(path):
    out = subprocess.check_output(["pip", "freeze"], text=True)
    Path(path).write_text(out)
    mlflow.log_artifact(str(path))
