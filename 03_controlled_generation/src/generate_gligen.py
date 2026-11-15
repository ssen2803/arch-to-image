"""ניסוי A/B (#14): GLIGEN עם תיבה וביטוי טקסט לכל אלמנט, במקום מסכה ב-ControlNet.

ControlNet-seg מכיר רק את 150 המחלקות של ADE20K, ולכן טאבון, מוקד אש ואגן הופכים כולם ל-fireplace.
GLIGEN מקבל לכל תיבה ביטוי חופשי ("tabun oven", "plastered collecting vat"), כך שהרזולוציה של הסכמה
יכולה להגיע עד התמונה. השאלה: האם זה בא על חשבון הנאמנות המרחבית?

אותו פרוטוקול כמו בשלב 3: 5 הדוגמאות, seeds 0–3, סגנון צילום, אותו פרומפט ואותו negative.
הפלט נשמר כ-outputs/<example>/gligen_s<seed>.png, ולכן summarize.py מעריך אותו אוטומטית
ומשווה אותו בזוגות מול photo (ControlNet) ומול photo_textonly.

אילוצי 6GB: FP16 (קובצי fp16), xformers עם נפילה ל-SDPA, ו-CPU offloading, כמו ב-generate.py.
רק אלמנטים נראים מקבלים תיבה: לא מוסתרים, ולא אלמנטים שמיקומם לא ידוע (אלה נכנסים לפרומפט בלבד).

שימוש:
    python src/generate_gligen.py
    python src/generate_gligen.py --examples ex01_iron_age_room --seeds 0 --beta 0.3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import NEGATIVE, SIZE, build_prompt, is_blank, run_with_retry  # noqa: E402
from mapping import ROOT  # noqa: E402

sys.path.insert(0, str(ROOT.parent / "02_layout_pipeline" / "src"))
from layout_core import image_boxes, load_layout  # noqa: E402
from render_layout import visibility  # noqa: E402

GLIGEN_MODEL = "masterful/gligen-1-4-generation-text-box"
EXAMPLES = ["ex01_iron_age_room", "ex02_chalcolithic_pits", "ex03_byzantine_winepress",
            "ex04_hellenistic_two_rooms", "ex05_roman_courtyard"]


def build_gligen(use_xformers: bool = True):
    import torch
    from diffusers import StableDiffusionGLIGENPipeline

    if not torch.cuda.is_available():
        raise SystemExit("לא נמצא GPU עם CUDA.")
    pipe = StableDiffusionGLIGENPipeline.from_pretrained(GLIGEN_MODEL, variant="fp16", dtype=torch.float16)
    attention = "sdpa"
    if use_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            attention = "xformers"
        except Exception as exc:
            print(f"אזהרה: xformers לא זמין ({type(exc).__name__}). ממשיך עם SDPA.")
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    return pipe, {"model": GLIGEN_MODEL, "attention": attention, "offload": "model", "dtype": "float16"}


def gligen_inputs(layout: dict) -> tuple[list[str], list[list[float]]]:
    """ביטוי ותיבה מנורמלת לכל אלמנט נראה. GLIGEN תומך בעד 30 תיבות."""
    visible = {k for k, v in visibility(layout).items() if v["status"] == "visible"}
    boxes = image_boxes(layout, SIZE, include=visible)[:30]
    return [b["phrase"] for b in boxes], [b["box"] for b in boxes]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="GLIGEN: תיבה וביטוי לכל אלמנט")
    ap.add_argument("--examples", nargs="+", default=EXAMPLES)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--beta", type=float, default=1.0,
                    help="gligen_scheduled_sampling_beta: חלק הצעדים שבהם התיבות פעילות (1 = כולם)")
    ap.add_argument("--tag", default="", help="סיומת לשם התנאי, למשל beta03")
    ap.add_argument("--only-blank", action="store_true", help="לייצר מחדש רק תמונות קיימות שיצאו ריקות")
    args = ap.parse_args()

    import torch
    pipe, info = build_gligen()
    cond = "gligen" + (f"_{args.tag}" if args.tag else "")
    for name in args.examples:
        layout = load_layout(ROOT.parent / "02_layout_pipeline" / "examples" / name / "layout.gold.json")
        phrases, boxes = gligen_inputs(layout)
        prompt = build_prompt(layout, "photo", None)
        out_dir = ROOT / "outputs" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = {"pipeline": info, "prompt": prompt, "negative": NEGATIVE, "phrases": phrases, "boxes": boxes,
                "steps": args.steps, "guidance": args.guidance, "beta": args.beta, "seeds": {}}
        meta_path = out_dir / f"{cond}_meta.json"
        if args.only_blank and meta_path.exists():
            meta["seeds"] = json.loads(meta_path.read_text(encoding="utf-8"))["seeds"]
        for seed in args.seeds:
            path = out_dir / f"{cond}_s{seed}.png"
            if args.only_blank:
                from PIL import Image
                if not path.exists() or not is_blank(Image.open(path)):
                    continue
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            image, used_seed = run_with_retry(
                pipe, seed, prompt=prompt, negative_prompt=NEGATIVE, gligen_phrases=phrases, gligen_boxes=boxes,
                gligen_scheduled_sampling_beta=args.beta, height=SIZE, width=SIZE,
                num_inference_steps=args.steps, guidance_scale=args.guidance)
            image.save(path)
            meta["seeds"][str(seed)] = {"used_seed": used_seed, "seconds": round(time.perf_counter() - t0, 1),
                                        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20)}
            print(f"{name} {cond}_s{seed}: {meta['seeds'][str(seed)]}", flush=True)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
