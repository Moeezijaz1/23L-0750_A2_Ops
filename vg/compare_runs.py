"""Task 3.5 - one comparison table + chart. Usage: python -m vg.compare_runs"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402

from vg import tracking  # noqa: E402
from vg.config import REPORT_DIR  # noqa: E402

COLS = ["tags.mlflow.runName", "params.model", "params.backbone", "params.lr", "params.img_size",
        "params.no_augment", "metrics.best_epoch", "metrics.val_f1", "metrics.val_roc_auc",
        "metrics.test_accuracy", "metrics.test_f1", "metrics.test_roc_auc", "metrics.test_recall",
        "metrics.test_precision"]

if __name__ == "__main__":
    tracking.setup()
    exp = mlflow.get_experiment_by_name(os.getenv("MLFLOW_EXPERIMENT_NAME", "VisionGuard"))
    df = mlflow.search_runs([exp.experiment_id], order_by=["metrics.val_roc_auc DESC"])
    df = df[[c for c in COLS if c in df.columns]].round(4)
    REPORT_DIR.mkdir(exist_ok=True)
    df.to_csv(REPORT_DIR / "run_comparison.csv", index=False)
    print(df.to_string(index=False))
    ax = df.set_index("tags.mlflow.runName")[["metrics.val_f1", "metrics.val_roc_auc"]].plot.barh(figsize=(8, 5))
    ax.set_title("Runs ranked by VALIDATION metrics")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "run_comparison.png", dpi=130)
