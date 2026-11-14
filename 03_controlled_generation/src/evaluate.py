"""הערכת הזיות על תמונות שנוצרו: HOR, Object Recall, HAR. הגדרות: measurement_HOR.md.

שימוש:
    # כל התמונות בתיקיית פלט של generate.py, מול ה-Layout שממנו נוצרו
    python src/evaluate.py --layout ../02_layout_pipeline/examples/ex01_iron_age_room/layout.gold.json \
                           --images outputs/ex01_iron_age_room/schematic_s*.png

    # בדיקת שפיות: מסכת ה-GT עצמה דרך מעריך הפלטה חייבת לתת HOR=0, recall=1
    python src/evaluate.py --layout .../layout.gold.json --images .../layout.gold.seg.png --evaluator palette

פלט: לכל תמונה <name>.hor.json ו-<name>.hor.png (אדום = מוזה, צהוב = חסר), וסיכום על כל התמונות.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hallucination import (  # noqa: E402
    HallucinationConfig, PaletteEvaluator, SegformerEvaluator, hallucination_report, layout_targets, overlay,
)
from mapping import load_mapping, project_labels_from_rgb  # noqa: E402


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="מדידת Hallucinated Object Rate")
    ap.add_argument("--layout", type=Path, required=True, help="ה-Layout שממנו נוצרו התמונות")
    ap.add_argument("--images", type=Path, nargs="+", required=True)
    ap.add_argument("--evaluator", choices=["segformer", "palette"], default="segformer")
    ap.add_argument("--segformer", default="nvidia/segformer-b4-finetuned-ade-512-512")
    ap.add_argument("--map", nargs="*", default=[], metavar="CLASS=ADE",
                    help="אותן דריסות מיפוי שבהן נוצרו התמונות")
    ap.add_argument("--min-area", type=int, default=HallucinationConfig.min_area_px)
    ap.add_argument("--dilate", type=int, default=HallucinationConfig.dilate_px)
    ap.add_argument("--granularity", choices=["fine", "coarse"], default="coarse",
                    help="coarse = בנוי מול קרקע (הרזולוציה שעברה כיול עם SegFormer בסגנון photo)")
    args = ap.parse_args()

    from layout_core import load_layout
    from render_layout import render_seg
    layout = load_layout(args.layout)
    gt = project_labels_from_rgb(np.array(render_seg(layout, 512)))
    targets = layout_targets(layout, 512)
    cfg = HallucinationConfig(min_area_px=args.min_area, dilate_px=args.dilate, granularity=args.granularity)
    mapping = load_mapping(overrides=dict(kv.split("=", 1) for kv in args.map))
    evaluator = PaletteEvaluator() if args.evaluator == "palette" else SegformerEvaluator(mapping, args.segformer)

    rows = []
    for path in args.images:
        image = np.array(Image.open(path).convert("RGB").resize((512, 512), Image.NEAREST))
        report = hallucination_report(evaluator(image), gt, targets["elements"], cfg, targets["valid"],
                                      targets["uncertain"], targets["occluded"])
        report.update({"image": str(path), "layout": str(args.layout), "evaluator": args.evaluator})
        stem = path.with_suffix("")
        Path(f"{stem}.hor.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        overlay(image, report).save(f"{stem}.hor.png")
        rows.append(report)
        print(f"{path.name}: HOR={report['HOR']:.3f} ({report['n_hallucinated']} מוזים, "
              f"{report['n_correct']} נכונים, foreign {report['foreign_components']}) "
              f"| HOR_area={report['HOR_area']:.3f} | recall={report['object_recall']:.3f} | HAR={report['HAR']:.3f}")

    if len(rows) > 1:
        for key in ("HOR", "object_recall", "HAR"):
            vals = [r[key] for r in rows]
            print(f"  {key}: mean {statistics.mean(vals):.3f}, median {statistics.median(vals):.3f}, n={len(vals)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
