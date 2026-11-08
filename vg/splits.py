"""Create leak-free stratified train/val/test lists (70/15/15).
Duplicates and unreadable files are dropped first, so identical images can never sit in two splits.
Usage: python -m vg.splits"""
import pandas as pd
from sklearn.model_selection import train_test_split

from vg.config import REPORT_DIR, SEED, SPLIT_DIR


def main():
    df = pd.read_csv(REPORT_DIR / "audit.csv")
    df = df[df.readable & ~df.duplicate & ~df.cross_class_dup & df.cls.notna()].copy()
    df["label"] = (df.cls == "ai").astype(int)
    train, rest = train_test_split(df, test_size=0.30, stratify=df.label, random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest.label, random_state=SEED)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part[["path", "label"]].to_csv(SPLIT_DIR / f"{name}.csv", index=False)
        print(name, len(part), part.label.value_counts().to_dict())


if __name__ == "__main__":
    main()
