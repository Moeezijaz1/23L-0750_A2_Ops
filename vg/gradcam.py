"""Bonus - Grad-CAM for the MobileNetV2 transfer model.
Usage: python -m vg.gradcam --model models/r3_mnv2_finetune_final.keras --threshold 0.5"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402

from vg.config import IMG_SIZE, REPORT_DIR  # noqa: E402
from vg.data import load_split  # noqa: E402
from vg.models import get_base  # noqa: E402
from vg.preprocess import decode_resize  # noqa: E402


def heatmap(model, img):
    """For MobileNetV2 the backbone output IS the last conv feature map (out_relu), so we take the
    gradient of the AI score with respect to that map and weight its channels (Grad-CAM)."""
    base = get_base(model)
    idx = model.layers.index(base)
    pre = model.layers[1:idx]       # preprocessing layers between Input and backbone
    post = model.layers[idx + 1:]   # GAP -> Dropout -> Dense
    x = tf.expand_dims(img, 0)
    for lyr in pre:
        x = lyr(x)
    conv = base(x, training=False)
    with tf.GradientTape() as tape:
        tape.watch(conv)
        y = conv
        for lyr in post:
            y = lyr(y, training=False)
        score = y[:, 0]
    grads = tape.gradient(score, conv)
    w = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(w * conv, -1))[0]
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy(), float(score[0])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", required=True, help="run name whose *_test_preds.npz to use")
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    model = keras.models.load_model(a.model, compile=False)
    d = np.load(REPORT_DIR / f"{a.run}_test_preds.npz")
    y, p = d["y"], d["p"]
    paths, _ = load_split("test")
    pred = (p >= a.threshold).astype(int)
    right = np.where(pred == y)[0][:4]
    wrong = np.where(pred != y)[0][:4]
    fig, axes = plt.subplots(2, 8, figsize=(24, 6.5))
    for col, (i, tag) in enumerate([(i, "correct") for i in right] + [(i, "WRONG") for i in wrong]):
        img = decode_resize(tf.io.read_file(paths[i]), IMG_SIZE)
        cam, _ = heatmap(model, img)
        cam = tf.image.resize(cam[..., None], (IMG_SIZE, IMG_SIZE)).numpy()[..., 0]
        axes[0, col].imshow(img.numpy().astype("uint8"))
        axes[0, col].set_title(f"{tag}: true={'AI' if y[i] else 'real'} P(AI)={p[i]:.2f}", fontsize=9)
        axes[1, col].imshow(img.numpy().astype("uint8"))
        axes[1, col].imshow(cam, cmap="jet", alpha=0.45)
        for r in (0, 1):
            axes[r, col].axis("off")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "gradcam.png", dpi=120)
    print("saved reports/gradcam.png")
