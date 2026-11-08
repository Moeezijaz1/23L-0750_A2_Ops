"""Task 3.6 - separate script: load model BY ALIAS and classify one image.
Usage: python -m vg.load_champion path/to/image.jpg"""
import json
import os
import sys

import mlflow
import tensorflow as tf
from mlflow import MlflowClient

from vg import tracking
from vg.config import IMG_SIZE
from vg.preprocess import decode_resize

if __name__ == "__main__":
    tracking.setup()
    name = os.getenv("REGISTERED_MODEL_NAME", "VisionGuard")
    uri = f"models:/{name}@champion"
    model = mlflow.keras.load_model(uri)
    mv = MlflowClient().get_model_version_by_alias(name, "champion")
    meta = mlflow.artifacts.load_dict(f"runs:/{mv.run_id}/model_meta.json")
    img = decode_resize(tf.io.read_file(sys.argv[1]), meta.get("img_size", IMG_SIZE))
    p = float(model.predict(tf.expand_dims(img, 0), verbose=0)[0, 0])
    label = "AI-generated" if p >= meta["threshold"] else "real"
    print(json.dumps({"model": f"{name} v{mv.version}", "p_ai": round(p, 4), "threshold": meta["threshold"],
                      "label": label}, indent=2))
