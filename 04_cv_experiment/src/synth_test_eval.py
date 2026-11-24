"""הערכה על ה-test הסינתטי (60 תמונות מ-30 פריסות שלא שימשו לאימון או לבחירת checkpoint).

חשוב לקריאת התוצאות: התיוג כאן הוא המסכה שהוזנה ל-ControlNet, והוא רועש (results/eda). הציון אומר כמה המודל
משחזר את המסכה, ולא כמה הוא טוב על צילום אמיתי. ההערכה על צילומים אמיתיים היא ב-real_walls_eval.py.

פלט: results/metrics/synth_test.json, results/metrics/synth_test_confusion.json, results/predictions/synth_test/

שימוש:
    python src/synth_test_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES, NUM_CLASSES, PROJECT, RUNS, SYNTH_DIR  # noqa: E402
from datasets import drop_blank_images, read_manifest, split_by_layout3  # noqa: E402
import importlib.util  # noqa: E402

# גם בשלב 3 יש evaluate.py, והנתיב שלו קודם ב-sys.path. לכן טוענים את הקובץ של שלב 4 במפורש.
_spec = importlib.util.spec_from_file_location("evaluate_stage4", Path(__file__).resolve().parent / "evaluate.py")
_ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ev)
UNetPredictor, colorize = _ev.UNetPredictor, _ev.colorize
from metrics import coarsen_labels, confusion, iou_from_confusion, miou  # noqa: E402

OUT = PROJECT / "results" / "metrics"
PRED = PROJECT / "results" / "predictions" / "synth_test"


class SegFormer5:
    def __init__(self):
        from hallucination import SegformerEvaluator
        from mapping import load_mapping
        self.ev = SegformerEvaluator(load_mapping())

    def __call__(self, im):
        return self.ev(np.array(im.convert("RGB")))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    rows, _ = drop_blank_images(read_manifest(SYNTH_DIR))
    _, _, test = split_by_layout3(rows)
    from zeroshot import GroundedSAM
    models = {"majority_background": [lambda im: np.zeros((im.height, im.width), np.uint8)],
              "segformer_ade": [SegFormer5()], "grounded_sam": [GroundedSAM()],
              "unet_synth": [UNetPredictor(RUNS / f"base_s{s}" / "best.pt") for s in (0, 1, 2)]}
    lids = np.array([r["layout_id"] for r in test])
    uniq = np.unique(lids)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(uniq), (2000, len(uniq)))
    res, confs, boots = {"n_images": len(test), "n_layouts": len(uniq), "models": {}}, {}, {}
    for name, ms in models.items():
        seeds_cm, seeds_cc = [], []
        for k, m in enumerate(ms):
            cms, ccs = [], []
            for r in test:
                im = Image.open(r["image_path"]).convert("RGB")
                gt = np.array(Image.open(r["label_path"]))
                p = np.asarray(m(im))
                if p.shape != gt.shape:
                    p = np.array(Image.fromarray(p).resize(gt.shape[::-1], Image.NEAREST))
                cms.append(confusion(p, gt))
                ccs.append(confusion(coarsen_labels(p), coarsen_labels(gt), n=3))
                if k == 0:
                    (PRED / name).mkdir(parents=True, exist_ok=True)
                    q = np.where(p < NUM_CLASSES, p, 0).astype(np.uint8)
                    colorize(q).save(PRED / name / f"{Path(r['image']).stem}.png", optimize=True)
            seeds_cm.append(np.stack(cms))
            seeds_cc.append(np.stack(ccs))
        b = []
        for cms in seeds_cm:
            per = np.stack([cms[lids == u].sum(0) for u in uniq])
            b.append([miou(per[i].sum(0)) for i in idx])
        b = np.mean(b, axis=0)
        boots[name] = b
        ious = [iou_from_confusion(c.sum(0)) for c in seeds_cm]
        res["models"][name] = {
            "miou": round(float(np.mean([np.nanmean(i) for i in ious])), 4),
            "miou_ci95": [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)],
            "miou_per_seed": [round(float(np.nanmean(i)), 4) for i in ious],
            "iou_per_class": {c: round(float(v), 4) for c, v in zip(CLASSES, np.nanmean(ious, axis=0))},
            "coarse_miou": round(float(np.mean([miou(c.sum(0)) for c in seeds_cc])), 4)}
        confs[name] = np.mean([c.sum(0) for c in seeds_cm], axis=0).round().astype(int).tolist()
        print(name, res["models"][name], flush=True)
    d = boots["unet_synth"] - boots["grounded_sam"]
    res["paired"] = {"unet_synth - grounded_sam": {
        "delta_miou": round(res["models"]["unet_synth"]["miou"] - res["models"]["grounded_sam"]["miou"], 4),
        "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "synth_test.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (OUT / "synth_test_confusion.json").write_text(json.dumps({"rows": CLASSES, "cols": CLASSES + ["none"], **confs}, indent=1))
    print(json.dumps(res["paired"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
