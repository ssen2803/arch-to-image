"""גשר בין 5 המחלקות של הפרויקט לבין ADE20K (מרחב התוויות של ControlNet-seg).

- לצד ההתניה: המרת מסכה בפלטת הפרויקט למסכה בצבעי ADE20K.
- לצד ההערכה: טבלת חיפוש (LUT) שממפה כל אחת מ-150 מחלקות ADE20K
  לתווית פרויקט, או ל-FOREIGN אם אין לה מקבילה באונטולוגיה.
"""

from __future__ import annotations

import json
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STAGE2_SRC = ROOT.parent / "02_layout_pipeline" / "src"
sys.path.insert(0, str(STAGE2_SRC))

from layout_core import CLASSES as PROJECT_CLASSES  # noqa: E402  ["background","floor","wall","installation","pit"]
from layout_core import PALETTE as PROJECT_PALETTE  # noqa: E402

# תוויות הפרויקט: האינדקס ב-PROJECT_CLASSES, ועוד שתי תוויות מיוחדות להערכה
LABEL = {name: i for i, name in enumerate(PROJECT_CLASSES)}
FOREIGN = len(PROJECT_CLASSES)  # 5: מחלקה שאין לה מקום באונטולוגיה (אדם, רכב, עץ...)
UNKNOWN = 255                   # פיקסל שהמעריך לא הצליח לסווג (רק במעריך הפלטה)

PALETTE_PATH = ROOT / "data" / "ade20k_palette.json"
MAPPING_PATH = ROOT / "config" / "class_mapping.json"


@lru_cache(maxsize=1)
def ade20k() -> tuple[list[str], np.ndarray]:
    d = json.loads(PALETTE_PATH.read_text(encoding="utf-8"))
    return d["classes"], np.array(d["palette"], dtype=np.uint8)


def load_mapping(path: Path = MAPPING_PATH, overrides: dict | None = None) -> dict:
    """טוען את המיפוי, מחיל דריסות התניה (למשל {"pit": "sand"}) ומוודא עקביות."""
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    m["condition"] = {**m["condition"], **(overrides or {})}
    classes, _ = ade20k()
    for proj, ade in m["condition"].items():
        if ade not in classes:
            raise ValueError(f"מחלקת ADE20K לא קיימת: '{ade}' (עבור {proj})")
    # מחלקת ההתניה של כל מחלקת פרויקט חייבת להיות בקבוצת ההערכה שלה, ורק בה.
    groups = {k: list(v) for k, v in m["evaluation_groups"].items()}
    for proj, ade in m["condition"].items():
        for other, members in groups.items():
            if other != proj and ade in members:
                members.remove(ade)
                warnings.warn(f"'{ade}' הועברה מקבוצת {other} לקבוצת {proj} כי היא משמשת להתניה של {proj}")
        if ade not in groups[proj]:
            groups[proj].append(ade)
    m["evaluation_groups"] = groups
    return m


def ade_color(name: str) -> tuple[int, int, int]:
    classes, palette = ade20k()
    return tuple(int(v) for v in palette[classes.index(name)])


def project_labels_from_rgb(seg_rgb: np.ndarray, max_dist: float = 0.0) -> np.ndarray:
    """מסכה בפלטת הפרויקט -> מפת תוויות 0..4.

    max_dist=0 דורש התאמה מדויקת (המסכות של שלב 2 מכילות רק צבעי פלטה).
    ערך חיובי מאפשר צבע קרוב (למשל אחרי שמירה ב-JPEG); פיקסל רחוק מכל
    צבע מקבל UNKNOWN.
    """
    pal = np.array([PROJECT_PALETTE[c] for c in PROJECT_CLASSES], dtype=np.float32)
    flat = seg_rgb.reshape(-1, 3).astype(np.float32)
    d = np.linalg.norm(flat[:, None, :] - pal[None, :, :], axis=2)
    labels = d.argmin(axis=1).astype(np.uint8)
    labels[d.min(axis=1) > max_dist] = UNKNOWN
    return labels.reshape(seg_rgb.shape[:2])


def labels_to_ade_rgb(labels: np.ndarray, mapping: dict) -> np.ndarray:
    """מפת תוויות פרויקט -> תמונת התניה בצבעי ADE20K עבור ControlNet."""
    if (labels == UNKNOWN).any():
        raise ValueError("במסכה יש פיקסלים שאינם בפלטת הפרויקט")
    out = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for name, idx in LABEL.items():
        out[labels == idx] = ade_color(mapping["condition"][name])
    return out


def ade_to_project_lut(mapping: dict) -> np.ndarray:
    """LUT באורך 150: אינדקס ADE20K -> תווית פרויקט (ברירת מחדל FOREIGN)."""
    classes, _ = ade20k()
    lut = np.full(len(classes), FOREIGN, dtype=np.uint8)
    for proj, members in mapping["evaluation_groups"].items():
        for ade in members:
            lut[classes.index(ade)] = LABEL[proj]
    return lut
