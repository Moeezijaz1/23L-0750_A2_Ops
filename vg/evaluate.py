"""Task 2.5 - evaluation helpers. Threshold is chosen on VALIDATION data only."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn import metrics as skm  # noqa: E402


def predict(model, ds):
    y = np.concatenate([b for _, b in ds.as_numpy_iterator()])
    p = model.predict(ds, verbose=0).ravel()
    return y, p


def choose_threshold(y, p, mode="f1"):
    """f1: maximise F1 | youden: maximise TPR-FPR | recall95: highest threshold with recall>=0.95"""
    ts = np.linspace(0.05, 0.95, 91)
    if mode == "f1":
        scores = [skm.f1_score(y, p >= t) for t in ts]
    elif mode == "youden":
        scores = [skm.recall_score(y, p >= t) - ((p >= t) & (y == 0)).sum() / max((y == 0).sum(), 1) for t in ts]
    elif mode == "recall95":
        scores = [t if skm.recall_score(y, p >= t) >= 0.95 else -1 for t in ts]
    else:
        raise ValueError(mode)
    return float(ts[int(np.argmax(scores))])


def metric_table(y, p, t):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = skm.confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(threshold=t, accuracy=skm.accuracy_score(y, pred),
                precision=skm.precision_score(y, pred, zero_division=0),
                recall=skm.recall_score(y, pred), f1=skm.f1_score(y, pred),
                roc_auc=skm.roc_auc_score(y, p), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def plot_confusion(y, p, t, path, title="Confusion matrix (test)"):
    cm = skm.confusion_matrix(y, (p >= t).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], ["pred real", "pred AI"])
    ax.set_yticks([0, 1], ["true real", "true AI"])
    ax.set_title(f"{title}, t={t:.2f}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_roc(y, p, path, title="ROC (test)"):
    fpr, tpr, _ = skm.roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, label=f"AUC={skm.roc_auc_score(y, p):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set(xlabel="False positive rate (real called AI)", ylabel="True positive rate", title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
