"""בניית דאטהסט סינתטי לאימון: Layout אקראי -> מסכה (= תיוג) -> תמונת צילום מ-ControlNet (שלב 3).

לכל תמונה נשמרים גם מדדי ההזיה של שלב 3 (רזולוציה גסה), כי במקומות שבהם ControlNet
המציא משהו התיוג שגוי: המסכה אומרת "רקע", אבל בתמונה יש קיר. בזמן האימון אפשר לסנן
לפי המדדים האלה (--max-har / --min-recall ב-train_unet.py), וזה גם ניסוי בפני עצמו.

הריצה ניתנת לחידוש: תמונות שכבר קיימות מדולגות.

תמונות ריקות: מסנן הבטיחות של SD מחזיר לפעמים תמונה שחורה (2 מתוך 600 בריצה הראשונה).
תמונה כזו לא נשמרת. במקומה נעשה ניסיון חוזר עם seed = seed + 1000·ניסיון, ועמודת seed
ב-manifest שומרת את ה-seed שבו השתמשו בפועל. שם הקובץ נשאר לפי ה-seed המבוקש.
הדגל --fix-blank מוחק תמונות ריקות קיימות ואת השורות שלהן, ומייצר אותן מחדש.

שימוש:
    python src/build_synthetic.py --n-layouts 300 --seeds 0 1
    python src/build_synthetic.py --n-layouts 300 --seeds 0 1 --fix-blank
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
from datasets import is_blank_image  # noqa: E402
from metrics import agreement  # noqa: E402
from synth_layouts import generate_layout  # noqa: E402

from generate import NEGATIVE, SIZE, STYLES, build_pipeline  # noqa: E402  (שלב 3)
from hallucination import HallucinationConfig, SegformerEvaluator, hallucination_report  # noqa: E402
from mapping import labels_to_ade_rgb, load_mapping, project_labels_from_rgb  # noqa: E402
from render_layout import render_seg  # noqa: E402  (שלב 2)

LIGHTING = ["harsh midday sun", "soft overcast light", "late afternoon light", "even diffuse light"]
FIELDS = ["image", "label", "layout_id", "seed", "agree_miou", "built_iou", "HOR_area", "object_recall", "HAR",
          "foreign_components", "prompt"]
MAX_ATTEMPTS = 3
RETRY_OFFSET = 1000


def remove_blank(out: Path) -> list[str]:
    """מוחק תמונות ריקות ואת השורות שלהן ב-manifest, כדי שהלולאה הרגילה תייצר אותן מחדש."""
    manifest = out / "manifest.csv"
    if not manifest.exists():
        return []
    with open(manifest, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    blank = [r["image"] for r in rows
             if (out / r["image"]).exists() and is_blank_image(np.asarray(Image.open(out / r["image"]).convert("L")))]
    if blank:
        with open(manifest, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(r for r in rows if r["image"] not in blank)
        for name in blank:
            (out / name).unlink()
    return blank


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-layouts", type=int, default=300)
    ap.add_argument("--first", type=int, default=0, help="seed של ה-Layout הראשון")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1], help="seeds של הדיפוזיה לכל Layout")
    ap.add_argument("--out", type=Path, default=SYNTH_DIR)
    ap.add_argument("--fix-blank", action="store_true", help="מחיקת תמונות ריקות קיימות וייצור מחדש")
    args = ap.parse_args()

    import torch
    for sub in ("layouts", "labels", "images"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    if args.fix_blank:
        removed = remove_blank(args.out)
        print(f"נמחקו {len(removed)} תמונות ריקות: {', '.join(removed) or '-'}")
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
                for attempt in range(MAX_ATTEMPTS):
                    used_seed = seed + RETRY_OFFSET * attempt
                    result = pipe(prompt=prompt, negative_prompt=NEGATIVE, image=control, height=SIZE, width=SIZE,
                                  num_inference_steps=30, guidance_scale=7.5, controlnet_conditioning_scale=1.0,
                                  generator=torch.Generator(device="cpu").manual_seed(used_seed))
                    image = result.images[0]
                    nsfw = getattr(result, "nsfw_content_detected", None)
                    if not ((nsfw and nsfw[0]) or is_blank_image(np.array(image))):
                        break
                    print(f"{name}: תמונה ריקה עם seed {used_seed}, מנסה שוב")
                else:
                    print(f"{name}: ריקה גם אחרי {MAX_ATTEMPTS} ניסיונות; לא נשמרת")
                    continue
                image.save(args.out / "images" / name)
                pred = evaluator(np.array(image))
                rep = hallucination_report(pred, labels, cfg=cfg)
                writer.writerow({"image": f"images/{name}", "label": f"labels/{lid}.png", "layout_id": lid,
                                 "seed": used_seed, **agreement(pred, labels),
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
