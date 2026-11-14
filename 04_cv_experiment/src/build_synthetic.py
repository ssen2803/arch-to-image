"""בניית דאטהסט סינתטי לאימון: Layout אקראי -> מסכה (= תיוג) -> תמונת צילום מ-ControlNet (שלב 3).

לכל תמונה נשמרים גם מדדי ההזיה של שלב 3 (רזולוציה גסה), כי במקומות שבהם ControlNet
המציא משהו התיוג שגוי: המסכה אומרת "רקע", אבל בתמונה יש קיר. בזמן האימון אפשר לסנן
לפי המדדים האלה (--max-har / --min-recall ב-train_unet.py), וזה גם ניסוי בפני עצמו.

הריצה ניתנת לחידוש: תמונות שכבר קיימות מדולגות.

שימוש:
    python src/build_synthetic.py --n-layouts 300 --seeds 0 1
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SYNTH_DIR  # noqa: E402
from metrics import agreement  # noqa: E402
from synth_layouts import generate_layout  # noqa: E402

from generate import NEGATIVE, SIZE, STYLES, build_pipeline  # noqa: E402  (שלב 3)
from hallucination import HallucinationConfig, SegformerEvaluator, hallucination_report  # noqa: E402
from mapping import labels_to_ade_rgb, load_mapping, project_labels_from_rgb  # noqa: E402
from render_layout import render_seg  # noqa: E402  (שלב 2)

LIGHTING = ["harsh midday sun", "soft overcast light", "late afternoon light", "even diffuse light"]
FIELDS = ["image", "label", "layout_id", "seed", "agree_miou", "built_iou", "HOR_area", "object_recall", "HAR",
          "foreign_components", "prompt"]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-layouts", type=int, default=300)
    ap.add_argument("--first", type=int, default=0, help="seed של ה-Layout הראשון")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1], help="seeds של הדיפוזיה לכל Layout")
    ap.add_argument("--out", type=Path, default=SYNTH_DIR)
    args = ap.parse_args()

    import torch
    for sub in ("layouts", "labels", "images"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    manifest = args.out / "manifest.csv"
    done = set()
    if manifest.exists():
        with open(manifest, encoding="utf-8") as f:
            done = {r["image"] for r in csv.DictReader(f)}

    mapping = load_mapping()
    pipe, info = build_pipeline("model", use_xformers=True)
    evaluator = SegformerEvaluator(mapping)
    cfg = HallucinationConfig(granularity="coarse")
    print(f"pipeline: {info}")

    with open(manifest, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not done:
            writer.writeheader()
        t_start, n_new = time.perf_counter(), 0
        for i in range(args.first, args.first + args.n_layouts):
            layout = generate_layout(i)
            lid = layout["source"]["id"]
            (args.out / "layouts" / f"{lid}.json").write_text(json.dumps(layout, indent=1), encoding="utf-8")
            labels = project_labels_from_rgb(np.array(render_seg(layout, SIZE)))
            Image.fromarray(labels.astype(np.uint8)).save(args.out / "labels" / f"{lid}.png")  # מצב L: אינדקס מחלקה
            control = Image.fromarray(labels_to_ade_rgb(labels, mapping))
            rng = random.Random(i)
            prompt = (STYLES["photo"].format(context=layout["source"]["context"], period=layout["source"]["period"])
                      + ", " + rng.choice(LIGHTING))
            for seed in args.seeds:
                name = f"{lid}_s{seed}.png"
                if f"images/{name}" in done:
                    continue
                image = pipe(prompt=prompt, negative_prompt=NEGATIVE, image=control, height=SIZE, width=SIZE,
                             num_inference_steps=30, guidance_scale=7.5, controlnet_conditioning_scale=1.0,
                             generator=torch.Generator(device="cpu").manual_seed(seed)).images[0]
                image.save(args.out / "images" / name)
                pred = evaluator(np.array(image))
                rep = hallucination_report(pred, labels, cfg=cfg)
                writer.writerow({"image": f"images/{name}", "label": f"labels/{lid}.png", "layout_id": lid,
                                 "seed": seed, **agreement(pred, labels),
                                 "HOR_area": rep["HOR_area"], "object_recall": rep["object_recall"],
                                 "HAR": rep["HAR"], "foreign_components": rep["foreign_components"],
                                 "prompt": prompt})
                f.flush()
                n_new += 1
            if (i - args.first + 1) % 10 == 0:
                rate = (time.perf_counter() - t_start) / max(n_new, 1)
                print(f"{i - args.first + 1}/{args.n_layouts} layouts, {n_new} new images, {rate:.1f}s/image")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
