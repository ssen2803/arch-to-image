"""יצירת תמונה 512x512 ממסכת סגמנטציה: Stable Diffusion 1.5 + ControlNet-seg, מותאם ל-6GB VRAM.

אילוצי חומרה (ממומשים ב-build_pipeline):
  1. FP16    - כל המשקולות נטענות ב-float16 (variant="fp16": ההורדה עצמה בחצי גודל).
  2. xformers - attention חסכוני בזיכרון. אם אין wheel תואם, נופלים ל-SDPA של PyTorch
               (אותו חיסכון בפועל ב-PyTorch 2.x) ומדווחים על כך ב-meta.
  3. CPU offloading - enable_model_cpu_offload: כל תת-מודל (text encoder, UNet+ControlNet,
               VAE) עולה ל-GPU רק בזמן השימוש בו. "sequential" חוסך עוד יותר אך איטי פי כמה.

קלט: Layout (JSON של שלב 2) או מסכה בפלטת הפרויקט (seg.png של שלב 2).
פלט: לכל seed תמונה PNG, וגם תמונת ההתניה בצבעי ADE20K וקובץ meta עם זיכרון שיא וזמנים.

שימוש:
    python src/generate.py --layout ../02_layout_pipeline/examples/ex01_iron_age_room/layout.gold.json
    python src/generate.py --layout ../02_layout_pipeline/examples/*/layout.gold.json --seeds 0 1 2 --style photo
    python src/generate.py --mask my_seg.png --prompt "top-down plan of a stone house" --out outputs/custom
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mapping import ROOT, labels_to_ade_rgb, load_mapping, project_labels_from_rgb  # noqa: E402

BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNET_MODEL = "lllyasviel/control_v11p_sd15_seg"
SIZE = 512

STYLES = {
    # ההכרעה משלב 1: סכמה מקודדת-ראיות (התוצר הראשי)
    "schematic": ("top-down archaeological site plan drawing of {context}, {period}, orthographic view, "
                  "clean ink outlines, flat muted colours, technical illustration, high detail"),
    # ריצת השוואה בלבד (שלב 1: פוטוריאליזם רק כביקורת)
    "photo": ("top-down drone orthophoto of an archaeological excavation, {context}, {period}, "
              "exposed stone walls, packed soil, natural light, high detail"),
}
NEGATIVE = ("people, person, workers, tools, car, vehicle, trees, plants, furniture, modern building, "
            "sky, horizon, perspective view, text, letters, labels, watermark, blurry, lowres")


def build_pipeline(offload: str = "model", use_xformers: bool = True):
    import torch
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

    if not torch.cuda.is_available():
        raise SystemExit("לא נמצא GPU עם CUDA. בדוק שהותקנה גרסת PyTorch עם CUDA (ראו README).")

    # (1) FP16: משקולות בחצי דיוק, ומורידים ישירות את קובצי ה-fp16
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_MODEL, dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_MODEL, controlnet=controlnet, dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # (2) xformers, עם נפילה מבוקרת ל-SDPA
    attention = "sdpa"
    if use_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            attention = "xformers"
        except Exception as exc:  # ModuleNotFoundError / ValueError כשאין wheel תואם
            print(f"אזהרה: xformers לא זמין ({type(exc).__name__}: {exc}). ממשיך עם SDPA של PyTorch.")

    # (3) CPU offloading: אין לקרוא ל-pipe.to("cuda") כשהוא פעיל
    if offload == "model":
        pipe.enable_model_cpu_offload()
    elif offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        raise ValueError(f"offload לא מוכר: {offload}")
    # (VAE slicing לא נדרש: מייצרים תמונה אחת בכל קריאה, ופענוח 512x512 נכנס בזיכרון.)
    pipe.set_progress_bar_config(disable=True)

    info = {"base_model": BASE_MODEL, "controlnet": CONTROLNET_MODEL, "dtype": "float16",
            "attention": attention, "offload": offload, "gpu": torch.cuda.get_device_name(0),
            "gpu_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 2**20)}
    return pipe, info


def load_condition(layout_path: Path | None, mask_path: Path | None, mapping: dict):
    """מחזיר (תוויות פרויקט 512x512, תמונת התניה ADE20K, layout או None, שם)."""
    if layout_path:
        from layout_core import errors_only, load_layout, validate
        from render_layout import render_seg
        layout = load_layout(layout_path)
        errs = errors_only(validate(layout))
        if errs:
            raise SystemExit(f"{layout_path}: ה-Layout לא תקין ({len(errs)} שגיאות). הרץ את validate_layout.py.")
        seg = np.array(render_seg(layout, SIZE))
        name = layout["source"]["id"]
    else:
        layout = None
        img = Image.open(mask_path).convert("RGB")
        if img.size != (SIZE, SIZE):
            img = img.resize((SIZE, SIZE), Image.NEAREST)  # NEAREST: לא ליצור צבעי ביניים בגבולות
        seg = np.array(img)
        name = mask_path.stem
    labels = project_labels_from_rgb(seg, max_dist=0.0 if layout else 30.0)
    return labels, labels_to_ade_rgb(labels, mapping), layout, name


RETRY_OFFSET = 1000
MAX_ATTEMPTS = 3


def is_blank(image) -> bool:
    """תמונה שחורה או אחידה. מסנן הבטיחות של SD מחזיר תמונה שחורה כשהוא מסמן תוצאה, וזה קרה
    גם על תמונות חפירה תמימות: 2 מתוך 600 בנתונים הסינתטיים, ו-1 מתוך 20 בניסוי GLIGEN."""
    import numpy as np
    g = np.asarray(image.convert("L"), dtype=np.float32)
    return bool(g.max() < 10 or g.std() < 2.0)


def run_with_retry(pipe, seed: int, **kwargs):
    """מריץ את ה-pipeline. אם התמונה סומנה או ריקה, מנסה שוב עם seed + 1000·ניסיון.
    מחזיר (תמונה, ה-seed שבו השתמשו בפועל). אחרי MAX_ATTEMPTS ניסיונות זורק RuntimeError."""
    import torch
    for attempt in range(MAX_ATTEMPTS):
        used = seed + RETRY_OFFSET * attempt
        result = pipe(generator=torch.Generator(device="cpu").manual_seed(used), **kwargs)
        image = result.images[0]
        nsfw = getattr(result, "nsfw_content_detected", None)
        if not ((nsfw and nsfw[0]) or is_blank(image)):
            return image, used
        print(f"seed {used}: תמונה ריקה (מסנן הבטיחות), מנסה שוב")
    raise RuntimeError(f"התמונה ריקה גם אחרי {MAX_ATTEMPTS} ניסיונות (seed {seed})")


def build_prompt(layout: dict | None, style: str, prompt: str | None) -> str:
    if prompt:
        return prompt
    src = layout["source"] if layout else {"context": "an excavated building", "period": "ancient"}
    text = STYLES[style].format(context=src["context"], period=src["period"])
    # אלמנט שמיקומו לא ידוע לא נמצא במסכה, אבל הטקסט אומר שהוא קיים - לכן הוא נכנס לפרומפט.
    # אלמנט מוסתר (מתחת לאחר) לעולם לא נכנס: הוא לא אמור להיראות.
    floating = [e["phrase"] for e in (layout or {}).get("elements", []) if e.get("position") == "unspecified"]
    return text + (", with " + ", ".join(floating) if floating else "")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="SD1.5 + ControlNet-seg, מותאם ל-6GB VRAM")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--layout", type=Path, nargs="+", help="קובצי Layout מאומתים (שלב 2)")
    src.add_argument("--mask", type=Path, nargs="+", help="מסכות בפלטת הפרויקט")
    ap.add_argument("--style", choices=sorted(STYLES), default="schematic")
    ap.add_argument("--prompt", help="דריסת הפרומפט (אחרת נגזר מה-Layout)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--cond-scale", type=float, default=1.0,
                    help="controlnet_conditioning_scale. 0 = תנאי בסיס text-only: אותו מודל, seed ופרומפט, בלי מידע מרחבי")
    ap.add_argument("--map", nargs="*", default=[], metavar="CLASS=ADE",
                    help="דריסת מיפוי התניה, למשל pit=sand installation=oven")
    ap.add_argument("--offload", choices=["model", "sequential"], default="model")
    ap.add_argument("--no-xformers", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs")
    ap.add_argument("--tag", default="", help="שם תנאי נוסף לקובצי הפלט, למשל bgsand (לניסויי A/B)")
    args = ap.parse_args()
    # שם התנאי: סגנון + text-only (אם cond-scale=0) + תג. כך תנאים שונים לא דורסים זה את זה.
    condition = args.style + ("_textonly" if args.cond_scale == 0 else "") + (f"_{args.tag}" if args.tag else "")

    import torch
    mapping = load_mapping(overrides=dict(kv.split("=", 1) for kv in args.map))
    pipe, info = build_pipeline(args.offload, use_xformers=not args.no_xformers)
    print(f"pipeline: {info}")

    for path in args.layout or args.mask:
        labels, control_rgb, layout, name = load_condition(
            path if args.layout else None, path if args.mask else None, mapping)
        out_dir = args.out / name
        out_dir.mkdir(parents=True, exist_ok=True)
        control = Image.fromarray(control_rgb)
        control.save(out_dir / f"control_ade20k{'_' + args.tag if args.tag else ''}.png")
        prompt = build_prompt(layout, args.style, args.prompt)

        runs = []
        for seed in args.seeds:
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            image, used_seed = run_with_retry(
                pipe, seed, prompt=prompt, negative_prompt=NEGATIVE, image=control,
                height=SIZE, width=SIZE, num_inference_steps=args.steps,
                guidance_scale=args.guidance, controlnet_conditioning_scale=args.cond_scale)
            dt = time.perf_counter() - t0
            fname = f"{condition}_s{seed}.png"
            image.save(out_dir / fname)
            runs.append({"seed": seed, "used_seed": used_seed, "file": fname, "seconds": round(dt, 1),
                         "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20),
                         "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20)})
            print(f"{name} seed={seed}: {dt:.1f}s, VRAM peak {runs[-1]['peak_allocated_mb']} MB -> {out_dir / fname}")

        meta = {**info, "source": str(path), "condition": condition, "style": args.style, "prompt": prompt,
                "negative_prompt": NEGATIVE, "steps": args.steps, "guidance": args.guidance,
                "cond_scale": args.cond_scale, "condition_mapping": mapping["condition"], "size": SIZE, "runs": runs}
        (out_dir / f"{condition}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                                         encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
