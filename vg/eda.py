"""Task 1.3 - visual EDA. Usage: python -m vg.eda  (run vg.audit first)"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from vg.config import DATA_DIR, REPORT_DIR, SEED  # noqa: E402


def sample_grid(df, cls, n=16, cols=4):
    sub = df[(df.cls == cls) & df.readable & ~df.duplicate].sample(n, random_state=SEED)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    for ax, (_, r) in zip(axes.ravel(), sub.iterrows()):
        with Image.open(DATA_DIR / r.path) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(f"{r.width}x{r.height}", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"Random samples - class: {cls}")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / f"eda_samples_{cls}.png", dpi=130)
    plt.close(fig)


def histograms(df):
    ok = df[df.readable]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, col in zip(axes, ["width", "height", "file_size"]):
        for cls, g in ok.groupby("cls"):
            ax.hist(g[col], bins=40, alpha=0.55, label=cls)
        ax.set_title(col)
        ax.set_yscale("log")
        ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "eda_histograms.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    REPORT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(REPORT_DIR / "audit.csv")
    for c in ["ai", "real"]:
        sample_grid(df, c)
    histograms(df)
    print("saved figures to", REPORT_DIR)
