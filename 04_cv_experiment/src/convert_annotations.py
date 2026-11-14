"""המרת ייצוא תיוג (מסכות צבע) למסכות מחלקה של הפרויקט: PNG חד-ערוצי, 0..4, 255=ignore.

מותאם לייצוא "Segmentation mask 1.1" של CVAT (תיקיית SegmentationClass), אבל עובד עם כל כלי
שמייצא מסכות צבע לפי annotation/labels.json:
  - שחור (0,0,0) = רקע. ב-CVAT פיקסל שלא תויג יוצא שחור, ולכן אין צורך לתייג רקע.
  - צבעי המחלקות = פלטת הפרויקט (אותם צבעים כמו בשלבים 2-3).
  - מגנטה (255,0,255) = ignore: אנשים, ציוד, צל עמוק, מחוץ לשטח החפירה, אזור לא ודאי.
צבע שאינו ברשימה מדווח כשגיאה ולא מומר בשקט.

שימוש:
    python src/convert_annotations.py --src export/SegmentationClass --dst data/real/test/masks
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import IGNORE, ROOT  # noqa: E402

LABELS_JSON = ROOT / "annotation" / "labels.json"


def color_table() -> dict[tuple[int, int, int], int]:
    """labels.json הוא בפורמט התוויות של CVAT (שם + צבע). האינדקס נגזר מהשם לפי האונטולוגיה."""
    from common import CLASSES
    spec = json.loads(LABELS_JSON.read_text(encoding="utf-8"))
    table = {(0, 0, 0): CLASSES.index("background")}  # CVAT מייצא פיקסל לא מתויג כשחור
    for lab in spec:
        rgb = tuple(int(lab["color"][i:i + 2], 16) for i in (1, 3, 5))
        table[rgb] = IGNORE if lab["name"] == "ignore" else CLASSES.index(lab["name"])
    return table


def convert(rgb: np.ndarray, table: dict) -> tuple[np.ndarray, dict]:
    out = np.full(rgb.shape[:2], IGNORE, dtype=np.uint8)
    found = np.zeros(rgb.shape[:2], bool)
    for color, idx in table.items():
        m = (rgb == np.array(color, dtype=np.uint8)).all(axis=2)
        out[m] = idx
        found |= m
    unknown = {}
    if not found.all():
        cols, counts = np.unique(rgb[~found].reshape(-1, 3), axis=0, return_counts=True)
        unknown = {str(tuple(int(v) for v in c)): int(n) for c, n in zip(cols, counts)}
    return out, unknown


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    args = ap.parse_args()
    args.dst.mkdir(parents=True, exist_ok=True)
    table = color_table()
    bad = 0
    for p in sorted(args.src.glob("*.png")):
        out, unknown = convert(np.array(Image.open(p).convert("RGB")), table)
        if unknown:
            bad += 1
            print(f"{p.name}: צבעים לא מוכרים (הומרו ל-ignore): {unknown}")
        Image.fromarray(out).save(args.dst / p.name)
    print(f"הומרו {len(list(args.src.glob('*.png')))} מסכות, {bad} עם צבעים לא מוכרים")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
