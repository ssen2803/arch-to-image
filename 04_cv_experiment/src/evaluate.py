"""הערכת שני ה-Baselines על קבוצת מבחן אמיתית ומתויגת (data/real/test).

לכל תמונה: חיזוי של כל מודל, מטריצת בלבול (5 מחלקות + גסה), ותגיות קומפוזיציה מה-GT.
סיכום:
  - mIoU ברמת הדאטהסט + רווח סמך 95% (bootstrap על תמונות), IoU לכל מחלקה, mIoU גס
  - הפרש מזווג (U-Net פחות zero-shot) עם רווח סמך מאותם אינדקסי bootstrap
  - פרוסות לפי קומפוזיציה (לניסוי ההכללה): mIoU על התמונות שמכילות כל תגית
  - הסכמה בין מתייגים (אם ניתן --human-b): התקרה המעשית של המדד

שימוש:
    python src/evaluate.py --split data/real/test --unet runs/base/best.pt --zeroshot grounded_sam --out results/test
    python src/evaluate.py --split data/real/test --human-b data/real/test_annotator_b/masks --out results/agreement
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASS_COLORS, CLASSES, COARSE_NAMES, NUM_CLASSES, ROOT  # noqa: E402
from compositions import ALL_TAGS, composition_tags  # noqa: E402
from datasets import real_items, to_tensor  # noqa: E402
from metrics import (  # noqa: E402
    bootstrap_miou, coarsen_labels, confusion, iou_from_confusion, miou, paired_bootstrap_delta,
)

MAX_SIDE = 1024  # כל המודלים רואים את אותה תמונה מוקטנת; ההשוואה ל-GT ברזולוציה המקורית


class UNetPredictor:
    def __init__(self, ckpt: Path, short_side: int = 512):
        from train_unet import predict
        from unet import ResNet18UNet
        self._predict = predict
        self.model = ResNet18UNet(NUM_CLASSES, pretrained=False).cuda().eval()
        state = torch.load(ckpt, map_location="cuda", weights_only=True)  # משקולות וערכים פשוטים בלבד
        self.model.load_state_dict(state["model"])
        self.short_side, self.name = short_side, f"unet:{Path(ckpt).parent.name}"

    def __call__(self, image: Image.Image) -> np.ndarray:
        s = self.short_side / min(image.size)
        img = image.resize((max(32, round(image.width * s)), max(32, round(image.height * s))), Image.BILINEAR)
        pred = self._predict(self.model, to_tensor(np.array(img))[None].cuda())[0].cpu().numpy().astype(np.uint8)
        return pred


def colorize(label: np.ndarray) -> Image.Image:
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i, c in enumerate(CLASS_COLORS):
        lut[i] = c
    return Image.fromarray(lut[label])


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--unet", type=Path, nargs="*", default=[])
    ap.add_argument("--zeroshot", choices=["grounded_sam", "sam3"])
    ap.add_argument("--prompts-json", type=Path)
    ap.add_argument("--box-threshold", type=float, default=0.3)
    ap.add_argument("--human-b", type=Path, help="תיקיית מסכות של מתייג שני (אותם שמות קבצים)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allow-synthetic", action="store_true",
                    help="מאפשר להריץ על תיקייה שאינה data/real (בדיקת צנרת בלבד, לא לדיווח)")
    args = ap.parse_args()

    split = args.split.resolve()
    if "real" not in split.parts and not args.allow_synthetic:
        raise SystemExit("קבוצת המבחן חייבת להיות תמונות אמיתיות תחת data/real (ראו test_set_protocol.md). "
                         "להרצת בדיקה על נתונים אחרים: --allow-synthetic")
    items = real_items(split)
    args.out.mkdir(parents=True, exist_ok=True)

    models = {}
    for ck in args.unet:
        m = UNetPredictor(ck)
        models[m.name] = m
    if args.zeroshot:
        from zeroshot import load_zeroshot
        kw = {"box_threshold": args.box_threshold} if args.zeroshot == "grounded_sam" else {}
        zs = load_zeroshot(args.zeroshot, args.prompts_json, **kw)
        models[f"zeroshot:{zs.name}"] = zs
        (args.out / "zeroshot_config.json").write_text(json.dumps(zs.config, indent=2), encoding="utf-8")
    if args.human_b:
        models["human_b"] = None

    cms = {n: [] for n in models}
    cms_coarse = {n: [] for n in models}
    tags_per_image, rows = [], []
    for it in items:
        image = Image.open(it["image_path"]).convert("RGB")
        gt = np.array(Image.open(it["label_path"]))
        s = min(1.0, MAX_SIDE / max(image.size))
        work = image.resize((round(image.width * s), round(image.height * s)), Image.BILINEAR) if s < 1 else image
        tags = composition_tags(gt)
        tags_per_image.append(tags)
        row = {"id": it["id"], "tags": ";".join(sorted(tags))}
        for name, model in models.items():
            if name == "human_b":
                pred = np.array(Image.open(args.human_b / f"{it['id']}.png"))
            else:
                pred = model(work)
                pred = np.array(Image.fromarray(pred).resize((gt.shape[1], gt.shape[0]), Image.NEAREST))
                safe = name.replace(":", "_")
                (args.out / safe).mkdir(exist_ok=True)
                colorize(pred).save(args.out / safe / f"{it['id']}.png")
            cm = confusion(pred, gt)
            cms[name].append(cm)
            cms_coarse[name].append(confusion(coarsen_labels(pred), coarsen_labels(gt), n=3))
            row[f"{name}_miou"] = round(miou(cm), 4)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    results = {"split": str(split), "n_images": len(items), "models": {}}
    for name in models:
        point, lo, hi = bootstrap_miou(cms[name])
        cp, clo, chi = bootstrap_miou(cms_coarse[name])
        iou = iou_from_confusion(np.stack(cms[name]).sum(0))
        slices = {}
        for tag in ALL_TAGS:
            idx = [i for i, t in enumerate(tags_per_image) if tag in t]
            if idx:
                sub = np.stack([cms[name][i] for i in idx]).sum(0)
                slices[tag] = {"n": len(idx), "miou": round(miou(sub), 4),
                               "pit_iou": round(float(iou_from_confusion(sub)[4]), 4)}
        results["models"][name] = {
            "miou": round(point, 4), "miou_ci95": [round(lo, 4), round(hi, 4)],
            "iou_per_class": {c: (None if np.isnan(v) else round(float(v), 4)) for c, v in zip(CLASSES, iou)},
            "coarse_miou": round(cp, 4), "coarse_miou_ci95": [round(clo, 4), round(chi, 4)],
            "by_composition": slices,
        }
    unets = [n for n in models if n.startswith("unet:")]
    zss = [n for n in models if n.startswith("zeroshot:")]
    results["paired"] = {f"{u} - {z}": paired_bootstrap_delta(cms[u], cms[z]) for u in unets for z in zss}

    (args.out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(args.out / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    lines = [f"# תוצאות על {split.name} ({len(items)} תמונות)", "",
             "| מודל | mIoU [95% CI] | " + " | ".join(CLASSES) + f" | mIoU גס ({'/'.join(COARSE_NAMES)}) |",
             "|---|---|" + "---|" * len(CLASSES) + "---|"]
    for name, r in results["models"].items():
        lines.append(f"| {name} | {r['miou']:.3f} [{r['miou_ci95'][0]:.3f}, {r['miou_ci95'][1]:.3f}] | "
                     + " | ".join("—" if v is None else f"{v:.3f}" for v in r["iou_per_class"].values())
                     + f" | {r['coarse_miou']:.3f} |")
    for k, d in results["paired"].items():
        lines += ["", f"**{k}:** ΔmIoU = {d['delta']:+.3f}, 95% CI [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}], "
                      f"P(Δ≤0) = {d['p_delta_le_0']:.3f}"]
    (args.out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
