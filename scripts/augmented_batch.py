"""Task 2.1 figure: one augmented training batch. Run from repo root: python scripts/augmented_batch.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vg.config import REPORT_DIR  # noqa: E402
from vg.data import make_ds  # noqa: E402

x, y = next(iter(make_ds("train", training=True, batch=9).take(1)))
fig, axes = plt.subplots(3, 3, figsize=(8, 8))
for ax, img, lab in zip(axes.ravel(), x.numpy(), y.numpy()):
    ax.imshow(img.clip(0, 255).astype("uint8"))
    ax.set_title("AI" if lab else "real")
    ax.axis("off")
fig.suptitle("One augmented training batch (flip / rotate / zoom / contrast)")
fig.tight_layout()
REPORT_DIR.mkdir(exist_ok=True)
fig.savefig(REPORT_DIR / "augmented_batch.png", dpi=130)
print("saved reports/augmented_batch.png")
