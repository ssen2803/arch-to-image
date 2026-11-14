"""טעינת נתונים: סינתטי (לאימון) ואמיתי (dev/test), כולל אוגמנטציות.

שני טיפולים בזליגת תוויות של ControlNet (מקרה קצה 1):
  1. סינון ברמת תמונה: לפי מדדי ההסכמה שנשמרו ב-manifest (agree_miou / HAR), ראו filter_rows.
  2. רצועת ignore סביב גבולות מחלקות: הדיפוזיה "גולשת" בגבולות, ולכן הפיקסלים
     הקרובים לגבול אינם אמינים כתיוג ולא נכנסים ל-loss. זה מטפל בגלישה המקומית,
     והסינון מטפל בהזיות הגסות (קיר שלם שלא היה במסכה).
"""

from __future__ import annotations

import csv
import io
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy import ndimage

from common import IGNORE

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------- רשימות פריטים

def read_manifest(synth_dir: Path) -> list[dict]:
    with open(synth_dir / "manifest.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["image_path"] = synth_dir / r["image"]
        r["label_path"] = synth_dir / r["label"]
        for k in ("agree_miou", "built_iou", "HOR_area", "object_recall", "HAR"):
            r[k] = float(r[k]) if r.get(k) not in (None, "", "None") else None
    return rows


def split_by_layout(rows: list[dict], val_every: int = 10) -> tuple[list[dict], list[dict]]:
    """פיצול לפי Layout ולא לפי תמונה: כמה seeds של אותו Layout הם כמעט שכפול,
    ולכן כולם חייבים להיות באותו צד, אחרת יש דליפה מהאימון לוולידציה."""
    train, val = [], []
    for r in rows:
        idx = int(r["layout_id"].split("_")[-1])
        (val if idx % val_every == 0 else train).append(r)
    return train, val


def filter_rows(rows: list[dict], min_agree: float = 0.0, max_har: float = 1.0) -> list[dict]:
    """סינון זליגת תוויות ברמת תמונה. הספים נקבעים בכיול (README, סעיף זליגת תוויות)."""
    out = []
    for r in rows:
        if r["agree_miou"] is not None and r["agree_miou"] < min_agree:
            continue
        if r["HAR"] is not None and r["HAR"] > max_har:
            continue
        out.append(r)
    return out


def real_items(split_dir: Path) -> list[dict]:
    """data/real/<split>/images/*.{jpg,png} + masks/<same stem>.png (0..4, 255=ignore)."""
    items = []
    for img in sorted((split_dir / "images").glob("*")):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        mask = split_dir / "masks" / f"{img.stem}.png"
        if not mask.exists():
            raise FileNotFoundError(f"חסרה מסכה עבור {img.name}: {mask}")
        items.append({"image_path": img, "label_path": mask, "id": img.stem})
    return items


# ---------------------------------------------------------------- עיבוד

def boundary_ignore(label: np.ndarray, px: int) -> np.ndarray:
    """פיקסלים במרחק עד px מגבול בין מחלקות מקבלים IGNORE."""
    if px <= 0:
        return label
    size = 2 * px + 1
    lab = label.astype(np.int16)
    edge = ndimage.maximum_filter(lab, size=size) != ndimage.minimum_filter(lab, size=size)
    out = label.copy()
    out[edge & (label != IGNORE)] = IGNORE
    return out


def to_tensor(img: np.ndarray) -> torch.Tensor:
    x = (img.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class SegDataset(torch.utils.data.Dataset):
    def __init__(self, items: list[dict], train: bool, crop: int = 384, geometric_aug: bool = True,
                 boundary_px: int = 0, seed: int = 0):
        self.items, self.train, self.crop = items, train, crop
        self.geometric_aug, self.boundary_px = geometric_aug, boundary_px
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.items)

    def _photometric(self, img: Image.Image) -> Image.Image:
        """פער הדומיין סינתטי->אמיתי: תאורה, צבע, חדות ודחיסת JPEG משתנים בצילומי שטח."""
        rng = self.rng
        from PIL import ImageEnhance
        for enh in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
            img = enh(img).enhance(rng.uniform(0.7, 1.3))
        if rng.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.5)))
        if rng.random() < 0.4:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=rng.randint(40, 90))
            img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
        if rng.random() < 0.05:
            img = img.convert("L").convert("RGB")
        return img

    def __getitem__(self, i: int):
        it = self.items[i]
        img = Image.open(it["image_path"]).convert("RGB")
        lab = np.array(Image.open(it["label_path"]))
        if lab.shape != (img.height, img.width):
            raise ValueError(f"גודל מסכה לא תואם לתמונה: {it['image_path']}")
        lab = boundary_ignore(lab, self.boundary_px)
        if not self.train:
            return to_tensor(np.array(img)), torch.from_numpy(lab.astype(np.int64)), str(it["image_path"])

        rng, c = self.rng, self.crop
        s = rng.uniform(0.75, 1.25)
        w, h = max(c, round(img.width * s)), max(c, round(img.height * s))
        img = img.resize((w, h), Image.BILINEAR)
        lab = np.array(Image.fromarray(lab).resize((w, h), Image.NEAREST))
        x0, y0 = rng.randint(0, w - c), rng.randint(0, h - c)
        img = img.crop((x0, y0, x0 + c, y0 + c))
        lab = lab[y0:y0 + c, x0:x0 + c]
        img = self._photometric(img)
        arr = np.array(img)
        if self.geometric_aug:  # כבוי בניסוי קומפוזיציות כיווניות (compositions.py)
            k = rng.randint(0, 3)
            arr, lab = np.rot90(arr, k), np.rot90(lab, k)
            if rng.random() < 0.5:
                arr, lab = arr[:, ::-1], lab[:, ::-1]
        return to_tensor(arr), torch.from_numpy(np.ascontiguousarray(lab).astype(np.int64)), str(it["image_path"])
