"""כמה טוב כל ציון אוטומטי מנבא את הדירוג האנושי של התאמת התיוג (#3).

קורא את annotation/rating_sample/ratings.csv (הדירוג) ואת runs/crossfit/rating_key.csv (הציונים),
ומחשב AUC לכל ציון: האם הוא מדרג תמונות תקינות (דירוג 1–2) מעל תמונות לא תואמות (דירוג 0).
בנוסף, רווח סמך של 95% ב-bootstrap, כי 40 תמונות הן מדגם קטן.

שימוש:
    python src/rating_auc.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RATINGS = ROOT / "annotation" / "rating_sample" / "ratings.csv"
KEY = ROOT / "runs" / "crossfit" / "rating_key.csv"
SCORES = {"crossfit_mean": "סינון בהצלבה", "segformer_agree": "SegFormer"}


def auc(scores: np.ndarray, good: np.ndarray) -> float:
    """הסתברות שתמונה תקינה אקראית מקבלת ציון גבוה מתמונה לא תואמת אקראית (תיקו = חצי)."""
    pos, neg = scores[good], scores[~good]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0) + 0.5 * (diff == 0)).mean())


def bootstrap_ci(scores: np.ndarray, good: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(scores), len(scores))
        v = auc(scores[idx], good[idx])
        if not np.isnan(v):
            vals.append(v)
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    with open(RATINGS, encoding="utf-8") as f:
        rated = {int(r["item"]): int(r["rating"]) for r in csv.DictReader(f) if r["rating"].strip()}
    if not rated:
        print(f"אין עדיין דירוגים ב-{RATINGS}")
        return 1
    with open(KEY, encoding="utf-8") as f:
        key = {int(r["item"]): r for r in csv.DictReader(f)}
    items = sorted(rated)
    good = np.array([rated[i] > 0 for i in items])
    print(f"דורגו {len(items)} תמונות: {int((~good).sum())} לא תואמות, {int(good.sum())} תקינות")
    for col, name in SCORES.items():
        s = np.array([float(key[i][col]) for i in items])
        lo, hi = bootstrap_ci(s, good)
        print(f"{name}: AUC = {auc(s, good):.3f}  (95% CI {lo:.3f}–{hi:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
