"""מדדי סגמנטציה: מטריצת בלבול, IoU לכל מחלקה, mIoU, ורווחי סמך ב-bootstrap.

mIoU מחושב ברמת הדאטהסט (סכימת מטריצות בלבול על כל התמונות, ואז IoU), כמקובל
ב-ADE20K/Cityscapes. ממוצע של mIoU לכל תמונה מוטה לטובת תמונות עם מעט מחלקות.
פיקסלים עם IGNORE (255) לא נספרים.
"""

from __future__ import annotations

import numpy as np

from common import COARSE_OF, IGNORE, NUM_CLASSES


def confusion(pred: np.ndarray, gt: np.ndarray, n: int = NUM_CLASSES) -> np.ndarray:
    """מטריצה n x n: שורות = GT, עמודות = חיזוי. חיזוי מחוץ לטווח (למשל FOREIGN) נספר כ'אף מחלקה'."""
    valid = gt != IGNORE
    g = gt[valid].astype(np.int64)
    p = pred[valid].astype(np.int64)
    inside = p < n
    cm = np.bincount(g[inside] * n + p[inside], minlength=n * n).reshape(n, n)
    # חיזוי מחוץ לטווח: FN למחלקת ה-GT, בלי FP לאף מחלקה
    out = np.bincount(g[~inside], minlength=n)
    cm_out = np.zeros((n, n + 1), dtype=np.int64)
    cm_out[:, :n] = cm
    cm_out[:, n] = out
    return cm_out


def iou_from_confusion(cm: np.ndarray) -> np.ndarray:
    """IoU לכל מחלקה. NaN אם המחלקה לא מופיעה לא ב-GT ולא בחיזוי."""
    n = cm.shape[0]
    tp = np.diag(cm[:, :n]).astype(float)
    fn = cm.sum(axis=1) - tp
    fp = cm[:, :n].sum(axis=0) - tp
    union = tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, tp / union, np.nan)


def miou(cm: np.ndarray) -> float:
    return float(np.nanmean(iou_from_confusion(cm)))


def coarsen_labels(labels: np.ndarray) -> np.ndarray:
    """5 מחלקות -> 3 (קרקע / בנוי / בור). IGNORE ותוויות מחוץ לטווח נשמרות."""
    lut = np.arange(256, dtype=np.uint8)
    for src, dst in COARSE_OF.items():
        lut[src] = dst
    for v in range(NUM_CLASSES, 256):
        lut[v] = v if v == IGNORE else 254  # כל תווית זרה -> "אף מחלקה" ברזולוציה הגסה
    return lut[labels]


def agreement(pred: np.ndarray, label: np.ndarray) -> dict:
    """הסכמה בין מסכת התיוג לבין מה שמעריך רואה בתמונה (לסינון נתונים סינתטיים)."""
    cm = confusion(coarsen_labels(pred), coarsen_labels(label), n=3)
    ious = iou_from_confusion(cm)
    return {"agree_miou": round(float(np.nanmean(ious)), 4),
            "built_iou": round(float(ious[1]), 4) if not np.isnan(ious[1]) else None}


def bootstrap_miou(cms: list[np.ndarray], n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """mIoU ורווח סמך 95%: דוגמים תמונות עם החזרה ומחשבים mIoU ברמת הדאטהסט."""
    rng = np.random.default_rng(seed)
    stack = np.stack(cms)
    point = miou(stack.sum(0))
    boots = [miou(stack[rng.integers(0, len(cms), len(cms))].sum(0)) for _ in range(n_boot)]
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def paired_bootstrap_delta(cms_a: list[np.ndarray], cms_b: list[np.ndarray], n_boot: int = 2000,
                           seed: int = 0) -> dict:
    """הפרש mIoU (A פחות B) על אותן תמונות, עם אותם אינדקסים של bootstrap לשני המודלים."""
    assert len(cms_a) == len(cms_b)
    rng = np.random.default_rng(seed)
    a, b = np.stack(cms_a), np.stack(cms_b)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        deltas.append(miou(a[idx].sum(0)) - miou(b[idx].sum(0)))
    deltas = np.array(deltas)
    return {"delta": miou(a.sum(0)) - miou(b.sum(0)),
            "ci95": (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))),
            "p_delta_le_0": float((deltas <= 0).mean())}
