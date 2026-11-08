"""Download the champion ONCE and store it on disk, so the server never calls DagsHub per request.
Usage: python -m vg.export_champion   -> models/champion.keras + models/champion_meta.json"""
import json
import os
import shutil
import tempfile
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from vg import tracking
from vg.config import MODEL_DIR

if __name__ == "__main__":
    tracking.setup()
    name = os.getenv("REGISTERED_MODEL_NAME", "VisionGuard")
    client = MlflowClient()
    mv = client.get_model_version_by_alias(name, "champion")
    tmp = tempfile.mkdtemp()
    local = mlflow.artifacts.download_artifacts(f"runs:/{mv.run_id}/model", dst_path=tmp)
    keras_files = list(Path(local).rglob("*.keras")) or list(Path(local).rglob("*.h5"))
    if not keras_files:
        raise SystemExit(f"no .keras file found under {local}")
    MODEL_DIR.mkdir(exist_ok=True)
    shutil.copy(keras_files[0], MODEL_DIR / "champion.keras")
    meta = mlflow.artifacts.load_dict(f"runs:/{mv.run_id}/model_meta.json")
    meta.update(model_name=name, model_version=str(mv.version), run_id=mv.run_id)
    (MODEL_DIR / "champion_meta.json").write_text(json.dumps(meta, indent=2))
    print("exported", name, "v" + str(mv.version), meta)
