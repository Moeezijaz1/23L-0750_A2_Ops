"""Create leak-free stratified train/val/test lists (70/15/15).
Duplicates and unreadable files are dropped first, so identical images can never sit in two splits.
Uses only numpy/pandas (no scikit-learn), so it also runs on locked-down Windows PCs.
Usage: python -m vg.splits"""
import numpy as np
import pandas as pd

from vg.config import REPORT_DIR, SEED, SPLIT_DIR


def stratified_split(df, fractions=(0.70, 0.15, 0.15), seed=SEED):
    """Shuffle each class separately and cut it at the same proportions, so every split keeps the class ratio."""
    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    for _, g in df.groupby("label"):
        idx = rng.permutation(g.index.to_numpy())
        n_train = int(round(len(idx) * fractions[0]))
        n_val = int(round(len(idx) * fractions[1]))
        parts["train"].append(idx[:n_train])
        parts["val"].append(idx[n_train:n_train + n_val])
        parts["test"].append(idx[n_train + n_val:])
    return {k: df.loc[np.concatenate(v)].sample(frac=1, random_state=seed) for k, v in parts.items()}


def main():
    df = pd.read_csv(REPORT_DIR / "audit.csv")
    df = df[df.readable & ~df.duplicate & ~df.cross_class_dup & df.cls.notna()].copy()
    df["label"] = (df.cls == "ai").astype(int)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in stratified_split(df).items():
        part.assign(path=part.path.str.replace("\\", "/", regex=False))[["path", "label"]].to_csv(
            SPLIT_DIR / f"{name}.csv", index=False)
        print(name, len(part), part.label.value_counts().to_dict())


if __name__ == "__main__":
    main()
