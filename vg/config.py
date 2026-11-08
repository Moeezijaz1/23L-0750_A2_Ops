"""Central settings for training scripts. Serving code reads env vars separately (app/)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data" / "raw"))
SPLIT_DIR = ROOT / "data" / "splits"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"
SEED = int(os.getenv("SEED", "42"))
IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))
BATCH = int(os.getenv("BATCH", "32"))
CLASS_NAMES = ["real", "ai"]  # index 1 (AI-generated) is the positive class


def class_from_path(rel_path) -> str | None:
    """Label = folder name. Kaggle folders are 'AiArtData' and 'RealArt'."""
    parts = [p.lower() for p in Path(rel_path).parts[:-1]]
    for p in parts:
        if p.startswith("aiart") or p in {"ai", "fake", "ai_generated", "aigenerated"}:
            return "ai"
    for p in parts:
        if p.startswith("real"):
            return "real"
    return None
