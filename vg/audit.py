"""Task 1.2 - dataset audit. Usage: python -m vg.audit"""
import hashlib

import pandas as pd
from PIL import Image

from vg.config import DATA_DIR, REPORT_DIR, class_from_path

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def sha256(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def audit(root=DATA_DIR) -> pd.DataFrame:
    rows = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in EXTS:
            continue
        rel = p.relative_to(root)
        row = dict(path=str(rel), cls=class_from_path(rel), file_size=p.stat().st_size,
                   sha256=sha256(p), readable=True, width=None, height=None,
                   mode=None, format=None, error="")
        try:
            with Image.open(p) as im:
                im.verify()  # cheap structural check
            with Image.open(p) as im:
                im.load()    # full decode catches truncated files
                row.update(width=im.width, height=im.height, mode=im.mode, format=im.format)
        except Exception as e:  # noqa: BLE001 - any failure means "unreadable"
            row.update(readable=False, error=f"{type(e).__name__}: {e}")
        rows.append(row)
    df = pd.DataFrame(rows)
    ok = df[df.readable]
    df["duplicate"] = False
    df.loc[ok.index, "duplicate"] = ok.duplicated("sha256", keep="first")
    # same bytes in BOTH classes would be a labelling conflict
    g = ok.groupby("sha256")["cls"].nunique()
    df["cross_class_dup"] = df.sha256.isin(g[g > 1].index)
    return df


def main():
    REPORT_DIR.mkdir(exist_ok=True)
    df = audit()
    df.to_csv(REPORT_DIR / "audit.csv", index=False)
    ok = df[df.readable]
    lines = ["Images per class:", df.cls.value_counts(dropna=False).to_string(), "",
             "Min / median / max per class:"]
    for c in ["width", "height", "file_size"]:
        lines += [c, ok.groupby("cls")[c].agg(["min", "median", "max"]).to_string(), ""]
    lines += ["Colour modes:", ok["mode"].value_counts().to_string(), "",
              "Formats:", ok["format"].value_counts().to_string(), "",
              f"Corrupt/unreadable files: {int((~df.readable).sum())}",
              f"Exact duplicates (extra copies): {int(df.duplicate.sum())}",
              f"Hashes appearing in both classes: {int(df[df.cross_class_dup].sha256.nunique())}"]
    text = "\n".join(lines)
    (REPORT_DIR / "audit_summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
