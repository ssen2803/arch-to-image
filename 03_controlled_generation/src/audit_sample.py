"""ביקורת אנושית של מדד ההזיות (measurement_HOR.md סעיף 6.4, משימה #6).

שני מצבים:
1. בניית המדגם (ברירת מחדל, דורש GPU ל-SegFormer):
   30 תמונות בסגנון צילום, מרובדות לפי דוגמה ותנאי. לכל תמונה נשמר קובץ עם שלושה חלקים:
   התמונה, התמונה עם הרכיבים שסומנו כהזיה (צבע ומספר לכל רכיב), ומסכת ה-Layout.
   בנוסף נוצרים components.csv (שורה לכל רכיב: האם זו באמת הזיה) ו-missed.csv
   (שורה לכל תמונה: כמה הזיות המעריך פספס).
2. ניקוד (--score): precision ו-recall של הסימון האוטומטי מול המתייג, ו-Cohen's kappa
   כשיש שני מתייגים (העמודות real_a ו-real_b).

שימוש:
    python src/audit_sample.py
    python src/audit_sample.py --score
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hallucination import HallucinationConfig, SegformerEvaluator, hallucination_report, layout_targets  # noqa: E402
from mapping import ROOT, load_mapping  # noqa: E402

sys.path.insert(0, str(ROOT.parent / "02_layout_pipeline" / "src"))
from layout_core import PALETTE, load_layout  # noqa: E402
from render_layout import render_seg  # noqa: E402

OUT = ROOT / "audit"
EXAMPLES = ROOT.parent / "02_layout_pipeline" / "examples"
# 6 לכל דוגמה: 3 ControlNet, 2 GLIGEN, 1 text-only. text-only מוסיף תמונות עם מעט אלמנטים, כדי לבדוק
# גם את מקרה הקצה שבו המעריך צריך לומר "אין כמעט כלום".
PLAN = {"photo": [0, 1, 2], "gligen": [0, 3], "photo_textonly": [1]}
COLORS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
          (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255)]


def build() -> int:
    mapping = load_mapping()
    evaluator = SegformerEvaluator(mapping)
    cfg = HallucinationConfig(granularity="coarse")
    rng = np.random.default_rng(0)
    items = [(ex.name, cond, seed) for ex in sorted((ROOT / "outputs").iterdir()) if ex.is_dir()
             for cond, seeds in PLAN.items() for seed in seeds]
    order = rng.permutation(len(items))
    (OUT / "items").mkdir(parents=True, exist_ok=True)
    comp_rows, missed_rows, key_rows = [], [], []
    for n, j in enumerate(order, start=1):
        name, cond, seed = items[j]
        layout = load_layout(EXAMPLES / name / "layout.gold.json")
        image = np.array(Image.open(ROOT / "outputs" / name / f"{cond}_s{seed}.png").convert("RGB"))
        targets = layout_targets(layout, image.shape[0])
        from mapping import project_labels_from_rgb
        gt = project_labels_from_rgb(np.array(render_seg(layout, image.shape[0])))
        rep = hallucination_report(evaluator(image), gt, targets["elements"], cfg, valid=targets["valid"],
                                   uncertain=targets["uncertain"], occluded=targets["occluded"],
                                   return_component_map=True)
        cmap = rep.pop("component_map")
        over = image.copy()
        for k in range(1, len(rep["hallucinated"]) + 1):
            m = cmap == k
            over[m] = (0.45 * over[m] + 0.55 * np.array(COLORS[(k - 1) % len(COLORS)])).astype(np.uint8)
        over_img = Image.fromarray(over)
        d = ImageDraw.Draw(over_img)
        for k, h in enumerate(rep["hallucinated"], start=1):
            ys, xs = np.nonzero(cmap == k)
            cy, cx = (int(ys.mean()), int(xs.mean())) if len(ys) else (h["bbox"][1], h["bbox"][0])
            d.rectangle([cx - 9, cy - 8, cx + 11, cy + 8], fill=(0, 0, 0))
            d.text((cx - 6, cy - 6), str(k), fill=(255, 255, 255))
            comp_rows.append({"item": n, "component": k, "class": h["class"], "area_px": h["area"],
                              "real_a": "", "real_b": ""})
        S = image.shape[0]
        canvas = Image.new("RGB", (3 * S + 20, S + 26), "white")
        canvas.paste(Image.fromarray(image), (0, 26))
        canvas.paste(over_img, (S + 10, 26))
        canvas.paste(render_seg(layout, S), (2 * S + 20, 26))
        dd = ImageDraw.Draw(canvas)
        dd.text((4, 6), f"#{n:02d}   generated image", fill="black")
        dd.text((S + 14, 6), f"flagged as invented: {len(rep['hallucinated'])} numbered regions", fill="black")
        dd.text((2 * S + 24, 6), "what the layout asks for", fill="black")
        canvas.save(OUT / "items" / f"item_{n:02d}.png")
        missed_rows.append({"item": n, "missed_hallucinations_a": "", "missed_hallucinations_b": ""})
        key_rows.append({"item": n, "example": name, "condition": cond, "seed": seed, "HOR": rep["HOR"],
                         "HOR_area": rep["HOR_area"], "n_flagged": len(rep["hallucinated"])})
    for fname, rows in (("components.csv", comp_rows), ("missed.csv", missed_rows)):
        with open(OUT / fname, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (OUT / "key.json").write_text(json.dumps(key_rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(items)} תמונות, {len(comp_rows)} רכיבים מסומנים -> {OUT}")
    return 0


def kappa(a: list[int], b: list[int]) -> float:
    """Cohen's kappa לשני מתייגים על תוויות בינאריות."""
    a, b = np.array(a), np.array(b)
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def score() -> int:
    with open(OUT / "components.csv", encoding="utf-8") as f:
        comps = list(csv.DictReader(f))
    with open(OUT / "missed.csv", encoding="utf-8") as f:
        missed = list(csv.DictReader(f))
    for rater in ("a", "b"):
        rated = [int(c[f"real_{rater}"]) for c in comps if c[f"real_{rater}"].strip()]
        miss = [int(m[f"missed_hallucinations_{rater}"]) for m in missed if m[f"missed_hallucinations_{rater}"].strip()]
        if not rated:
            continue
        tp = sum(rated)
        precision = tp / len(rated)
        recall = tp / (tp + sum(miss)) if (tp + sum(miss)) else float("nan")
        print(f"מתייג {rater}: {len(rated)} רכיבים. precision = {precision:.3f}, recall = {recall:.3f} "
              f"({sum(miss)} הזיות שפוספסו ב-{len(miss)} תמונות)")
    both = [(int(c["real_a"]), int(c["real_b"])) for c in comps if c["real_a"].strip() and c["real_b"].strip()]
    if both:
        print(f"Cohen's kappa בין המתייגים: {kappa(*zip(*both)):.3f} ({len(both)} רכיבים)")
    if not any(c["real_a"].strip() for c in comps):
        print("אין עדיין דירוגים ב-components.csv")
        return 1
    return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    return score() if ap.parse_args().score else build()


if __name__ == "__main__":
    raise SystemExit(main())
