"""Task 2.2 table: output shape + parameter count per layer. Run: python scripts/model_table.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd  # noqa: E402

from vg.config import REPORT_DIR  # noqa: E402
from vg.models import build_baseline  # noqa: E402

m = build_baseline()
m.summary()  # screenshot this
rows = [dict(layer=lyr.name, type=type(lyr).__name__, output_shape=str(lyr.output.shape), params=lyr.count_params())
        for lyr in m.layers]
df = pd.DataFrame(rows)
REPORT_DIR.mkdir(exist_ok=True)
df.to_csv(REPORT_DIR / "baseline_layer_table.csv", index=False)
print(df.to_string(index=False))
print("total params:", m.count_params())
