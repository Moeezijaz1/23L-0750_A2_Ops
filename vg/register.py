"""Task 3.6 - register the best run's model, create versions, set alias 'champion'.
Usage: python -m vg.register --run-id <mlflow_run_id> [--alias champion]
Run it twice (two different runs) to get two versions; alias moves to whichever version you choose."""
import argparse
import os

import mlflow
from mlflow import MlflowClient

from vg import tracking

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--alias", default="champion")
    ap.add_argument("--no-alias", action="store_true")
    a = ap.parse_args()
    tracking.setup()
    name = os.getenv("REGISTERED_MODEL_NAME", "VisionGuard")
    mv = mlflow.register_model(f"runs:/{a.run_id}/model", name)
    client = MlflowClient()
    if not a.no_alias:
        client.set_registered_model_alias(name, a.alias, mv.version)
    print(f"registered {name} v{mv.version}" + ("" if a.no_alias else f" alias={a.alias}"))
