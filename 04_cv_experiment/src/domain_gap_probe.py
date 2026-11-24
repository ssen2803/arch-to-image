"""האם התאמות פשוטות בזמן הבדיקה מצילות את ה-U-Net שאומן רק על תמונות סינתטיות? על 14 אריחי dev בלבד.

  side       גודל הקלט (הקירות הסינתטיים עבים יותר ביחס לתמונה)
  saturation הגברת רוויה פי 2.5 (התמונות הסינתטיות רוויות יותר)
  blur       טשטוש קל, כדי להתקרב למראה החלק של הקירות הסינתטיים
  adabn      חישוב מחדש של סטטיסטיקות ה-BatchNorm על אריחי dev, בלי תוויות

פלט: results/metrics/domain_gap_probe.json

שימוש:
    python src/domain_gap_probe.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUNS  # noqa: E402
from datasets import real_items, to_tensor  # noqa: E402
from real_walls_eval import OUT, WALL, WALLS, UNet, counts, scores  # noqa: E402


def run(model, dev, side, sat=1.0, blur=0) -> float:
    total = []
    for it in dev:
        im = Image.open(it["image_path"]).convert("RGB")
        gt = np.array(Image.open(it["label_path"]))
        x = ImageEnhance.Color(im).enhance(sat)
        x = x.filter(ImageFilter.GaussianBlur(blur)) if blur else x
        k = side / min(im.size)
        x = x.resize((round(im.width * k / 32) * 32, round(im.height * k / 32) * 32), Image.BILINEAR)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            p = model(to_tensor(np.array(x))[None].cuda()).argmax(1)[0].cpu().numpy().astype(np.uint8)
        total.append(counts(np.array(Image.fromarray(p).resize(im.size, Image.NEAREST)) == WALL, gt))
    return round(float(scores(np.sum(total, 0))["iou"]), 4)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    dev = real_items(WALLS / "dev")
    out: dict = {"n_tiles": len(dev), "wall_iou_everything_wall": None, "seeds": {}}
    aw = [counts(np.ones(np.array(Image.open(i["label_path"])).shape, bool), np.array(Image.open(i["label_path"]))) for i in dev]
    out["wall_iou_everything_wall"] = round(float(scores(np.sum(aw, 0))["iou"]), 4)
    for s in (0, 1, 2):
        m = UNet(RUNS / f"base_s{s}" / "best.pt", 512).model
        r = {f"side={side} sat={sat} blur={b}": run(m, dev, side, sat, b)
             for side in (384, 512, 800, 1136) for sat in (1.0, 2.5) for b in (0, 3)}
        for side in (384, 512):
            a = UNet(RUNS / f"base_s{s}" / "best.pt", 512).model
            for mod in a.modules():
                if isinstance(mod, torch.nn.BatchNorm2d):
                    mod.reset_running_stats()
                    mod.momentum = None
                    mod.train()
            with torch.no_grad():
                for it in dev:
                    a(to_tensor(np.array(Image.open(it["image_path"]).convert("RGB").resize((side, side), Image.BILINEAR)))[None].cuda())
            a.eval()
            r[f"adabn side={side}"] = run(a, dev, side)
        out["seeds"][s] = r
        print(s, "max without adabn:", max(v for k, v in r.items() if "adabn" not in k), "adabn:", {k: v for k, v in r.items() if "adabn" in k}, flush=True)
    al = [v for r in out["seeds"].values() for k, v in r.items()]
    out["max_iou_any_setting"] = max(al)
    out["max_iou_without_adabn"] = max(v for r in out["seeds"].values() for k, v in r.items() if "adabn" not in k)
    (OUT / "domain_gap_probe.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
