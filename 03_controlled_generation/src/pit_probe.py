"""מה SegFormer רואה במקום שבו הפריסה מציבה בור: ספירת מחלקות ADE20K בתוך מסכות הבורות, לכל תנאי בפיילוט.

המדד אינו נותן קרדיט לבורות כמעט בכלל (0 מתוך 28 ב-ControlNet). הבדיקה מראה לאיזו מחלקת ADE20K הפיקסלים האלה
הולכים, ואינה מבודדת את הגורם (פרומפט, מיפוי, או המעריך עצמו).

פלט: results/metrics/pit_probe.json

שימוש:
    python src/pit_probe.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hallucination import LABEL, SegformerEvaluator, layout_targets  # noqa: E402
from mapping import ROOT, load_mapping  # noqa: E402
from layout_core import load_layout  # noqa: E402

CONDS = ("photo_textonly", "gligen", "photo", "photo_llm")
EXAMPLES = ROOT.parent / "02_layout_pipeline" / "examples"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ev = SegformerEvaluator(load_mapping())
    names = ev.model.config.id2label
    out = {}
    for c in CONDS:
        cnt, n, top = Counter(), 0, Counter()
        for d in sorted(p for p in (ROOT / "outputs").iterdir() if p.is_dir()):
            pits = [e["mask"] for e in layout_targets(load_layout(EXAMPLES / d.name / "layout.gold.json"), 512)["elements"] if e["label"] == LABEL["pit"]]
            for s in range(4):
                im = Image.open(d / f"{c}_s{s}.png").convert("RGB").resize((512, 512), Image.NEAREST)
                ade = ev.ade_labels(np.array(im)) if pits else None
                for m in pits:
                    v, k = np.unique(ade[m], return_counts=True)
                    cnt.update({names[int(a)]: int(b) for a, b in zip(v, k)})
                    top[names[int(v[k.argmax()])]] += 1
                    n += 1
        tot = sum(cnt.values())
        out[c] = {"pits": n, "pixel_share": {k: round(v / tot, 4) for k, v in cnt.most_common(8)}, "majority_class_per_pit": dict(top.most_common())}
    dst = ROOT.parent / "results" / "metrics" / "pit_probe.json"
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
