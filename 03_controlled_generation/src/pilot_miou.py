"""העדשה הראשונה של שאלת המחקר בפיילוט: mIoU מול פריסת הטקסט, לכל תנאי (אחרי סגמנטציה מחדש ב-SegFormer, רזולוציה גסה).

המקרה (הפריסה) הוא יחידת הניתוח: ממוצע על seeds, והפרש מזווג מול "טקסט בלבד" לכל מקרה. עם 5 מקרים התוצאה
תיאורית: מדווחים כל המקרים ו-p מדויק של היפוך סימנים (המינימום האפשרי הוא 1/16). ה-IoU לכל מחלקה (קרקע, בנוי, בור)
מסביר את הכיוון: פריסה מעלה את IoU של הבנוי ומורידה את IoU של הקרקע.

פלט: results/metrics/pilot_miou.json

שימוש:
    python src/pilot_miou.py
"""

from __future__ import annotations

import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fidelity import class_map, miou  # noqa: E402
from hallucination import SegformerEvaluator, layout_targets  # noqa: E402
from mapping import ROOT, load_mapping  # noqa: E402
from layout_core import load_layout  # noqa: E402

CONDS = ("photo_textonly", "gligen", "photo", "photo_llm")
EXAMPLES = ROOT.parent / "02_layout_pipeline" / "examples"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ev = SegformerEvaluator(load_mapping())
    cell, cls = defaultdict(list), defaultdict(list)
    for d in sorted(p for p in (ROOT / "outputs").iterdir() if p.is_dir()):
        gold = load_layout(EXAMPLES / d.name / "layout.gold.json")
        ref, valid = class_map(gold, 512), layout_targets(gold, 512)["valid"]
        for c in CONDS:
            for s in range(4):
                im = Image.open(d / f"{c}_s{s}.png").convert("RGB").resize((512, 512), Image.NEAREST)
                m = miou(ev(np.array(im)), ref, valid)
                cell[(c, d.name)].append(m["miou"])
                for k, name in ((0, "ground"), (2, "built"), (4, "pit")):
                    if k in m["per_class"]:
                        cls[(c, name, d.name)].append(m["per_class"][k])
    exs = sorted({e for _, e in cell})
    M = {c: np.array([np.mean(cell[(c, e)]) for e in exs]) for c in CONDS}
    signs = np.array(list(itertools.product([1, -1], repeat=len(exs))))
    out = {"n_cases": len(exs), "seeds": 4, "granularity": "coarse: ground (floor included), built, pit", "cases": exs, "conditions": {}}
    for c in CONDS:
        d = M[c] - M["photo_textonly"]
        out["conditions"][c] = {
            "miou": round(float(M[c].mean()), 4), "per_case": [round(float(v), 4) for v in M[c]],
            "minus_text_only": round(float(d.mean()), 4), "diffs": [round(float(v), 4) for v in d],
            "n_higher": int((d > 0).sum()),
            "p_flip": round(float((np.abs((signs * d).mean(1)) >= abs(d.mean()) - 1e-12).mean()), 4),
            "iou_by_class": {n: round(float(np.mean([np.mean(cls[(c, n, e)]) for e in exs if (c, n, e) in cls])), 4) for n in ("ground", "built", "pit")}}
    dst = ROOT.parent / "results" / "metrics" / "pilot_miou.json"
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
