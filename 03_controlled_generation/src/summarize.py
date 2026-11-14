"""הערכת כל התמונות ב-outputs/ וסיכום לפי תנאי, בשתי רזולוציות (fine / coarse).

ממצעים קודם על seeds ואחר כך על Layouts, כי ה-Layout הוא יחידת הניתוח (measurement_HOR.md סעיף 7).
המעריך רץ פעם אחת לכל תמונה, ושתי הרזולוציות מחושבות מאותה מפת תוויות.

שימוש:
    python src/summarize.py
    python src/summarize.py --conditions photo photo_textonly

פלט: outputs/summary.csv (שורה לכל תמונה ורזולוציה) ו-outputs/summary.md (טבלאות).
לכל תמונה נשמרים גם <name>.hor.json (שתי הרזולוציות) ו-<name>.hor.png (המחשה ברזולוציה הגסה).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hallucination import (  # noqa: E402
    HallucinationConfig, SegformerEvaluator, hallucination_report, layout_targets, overlay,
)
from mapping import ROOT, load_mapping, project_labels_from_rgb  # noqa: E402

EXAMPLES = ROOT.parent / "02_layout_pipeline" / "examples"
METRICS = ("HOR", "HOR_area", "object_recall", "HAR", "foreign_components")
LOWER_IS_BETTER = {"HOR": True, "HOR_area": True, "object_recall": False, "HAR": True, "foreign_components": True}
GRANULARITIES = ("fine", "coarse")


def condition_of(image: Path) -> str:
    return image.stem.rsplit("_s", 1)[0]  # photo_s0 -> photo ; photo_textonly_s2 -> photo_textonly


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    ap.add_argument("--conditions", nargs="*", help="ברירת מחדל: כל התנאים שנמצאו")
    args = ap.parse_args()

    from layout_core import load_layout
    from render_layout import render_seg

    evaluator = SegformerEvaluator(load_mapping())
    rows = []
    for out_dir in sorted(p for p in args.outputs.iterdir() if p.is_dir()):
        layout = load_layout(EXAMPLES / out_dir.name / "layout.gold.json")
        gt = project_labels_from_rgb(np.array(render_seg(layout, 512)))
        t = layout_targets(layout, 512)
        for img_path in sorted(out_dir.glob("*_s[0-9]*.png")):
            if img_path.name.endswith(".hor.png"):
                continue
            cond = condition_of(img_path)
            if args.conditions and cond not in args.conditions:
                continue
            image = np.array(Image.open(img_path).convert("RGB"))
            pred = evaluator(image)
            reports = {g: hallucination_report(pred, gt, t["elements"], HallucinationConfig(granularity=g), t["valid"],
                                          t["uncertain"], t["occluded"])
                       for g in GRANULARITIES}
            img_path.with_suffix(".hor.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2),
                                                         encoding="utf-8")
            overlay(image, reports["coarse"]).save(img_path.with_suffix(".hor.png"))
            seed = img_path.stem.rsplit("_s", 1)[1]
            for g, rep in reports.items():
                rows.append({"layout": out_dir.name, "condition": cond, "seed": seed, "granularity": g,
                             **{k: rep[k] for k in METRICS}})

    with open(args.outputs / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    conditions = sorted({r["condition"] for r in rows})
    layouts = sorted({r["layout"] for r in rows})
    n_images = len(rows) // len(GRANULARITIES)
    lines = [f"# סיכום הזיות (מעריך: {evaluator.model_id}, {n_images} תמונות)", "",
             "> פיילוט: 5 Layouts. המספרים תיאוריים, ולא מבחן מובהקות. בסגנון schematic, SegFormer "
             "מפרש את האיור כפוסטר או מסך (כשל תחום שזוהה בכיול), ולכן המספרים שם אינם תקפים.", ""]

    for g in GRANULARITIES:
        grow = [r for r in rows if r["granularity"] == g]
        per_layout = defaultdict(lambda: defaultdict(list))
        for r in grow:
            for k in METRICS:
                per_layout[(r["layout"], r["condition"])][k].append(r[k])
        lines += [f"## רזולוציה: {g}", "",
                  "| תנאי | HOR ↓ | HOR_area ↓ | Object Recall ↑ | HAR ↓ | רכיבים זרים ↓ |",
                  "|---|---|---|---|---|---|"]
        for c in conditions:
            means = {k: statistics.mean(statistics.mean(per_layout[(lay, c)][k]) for lay in layouts
                                        if (lay, c) in per_layout) for k in METRICS}
            flag = " ⚠ לא תקף" if c.startswith("schematic") else ""
            lines.append(f"| {c}{flag} | " + " | ".join(f"{means[k]:.3f}" for k in METRICS) + " |")

        key = {(r["layout"], r["seed"], r["condition"]): r for r in grow}
        comparisons = [(c, base) for base in ("photo_textonly", "photo") for c in conditions
                       if base in conditions and c != base and not c.startswith("schematic")
                       and not (base == "photo" and c == "photo_textonly")]
        for c, base in comparisons:
            pairs = [(key[(lay, s, c)], key[(lay, s, base)])
                     for (lay, s, cc) in key if cc == c and (lay, s, base) in key]
            lines += ["", f"השוואה מזווגת: **{c}** מול **{base}** (אותו Layout ואותו seed, {len(pairs)} זוגות). "
                          f"מספר הזוגות שבהם {c} טוב יותר:", ""]
            for k in ("HOR", "HOR_area", "object_recall", "HAR"):
                better = sum((a[k] < b[k]) if LOWER_IS_BETTER[k] else (a[k] > b[k]) for a, b in pairs)
                ties = sum(a[k] == b[k] for a, b in pairs)
                lines.append(f"- {k}: {better} מתוך {len(pairs)}" + (f" (שוויון: {ties})" if ties else ""))

        lines += ["", "לפי Layout (HOR_area / recall, ממוצע על seeds):", "",
                  "| Layout | " + " | ".join(conditions) + " |", "|---|" + "---|" * len(conditions)]
        for lay in layouts:
            cells = []
            for c in conditions:
                d = per_layout.get((lay, c))
                cells.append(f"{statistics.mean(d['HOR_area']):.2f} / {statistics.mean(d['object_recall']):.2f}"
                             if d else "—")
            lines.append(f"| {lay} | " + " | ".join(cells) + " |")
        lines.append("")

    (args.outputs / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
