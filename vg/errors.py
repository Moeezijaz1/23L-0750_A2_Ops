"""Task 2.6 - show 10 misclassified test images. Usage: python -m vg.errors --run r3_mnv2_finetune --threshold 0.45"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from vg.config import REPORT_DIR  # noqa: E402
from vg.data import load_split  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    d = np.load(REPORT_DIR / f"{a.run}_test_preds.npz")
    y, p = d["y"], d["p"]
    paths, _ = load_split("test")
    pred = (p >= a.threshold).astype(int)
    wrong = np.where(pred != y)[0]
    wrong = wrong[np.argsort(-np.abs(p[wrong] - a.threshold))][: a.n]  # most confident mistakes first
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    for ax, i in zip(axes.ravel(), wrong):
        with Image.open(paths[i]) as im:
            ax.imshow(im.convert("RGB"))
            size = f"{im.width}x{im.height}"
        ax.set_title(f"true={'AI' if y[i] else 'real'} pred={'AI' if pred[i] else 'real'}\nP(AI)={p[i]:.2f} {size}",
                     fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    out = REPORT_DIR / f"{a.run}_errors.png"
    fig.savefig(out, dpi=130)
    print("saved", out, "| total errors:", int((pred != y).sum()), "of", len(y))
    for i in wrong:
        print(i, paths[i], f"P(AI)={p[i]:.3f}")
