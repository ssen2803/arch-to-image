"""הערכה ראשונה על צילומים אמיתיים: סגמנטציית קירות (קיר או לא) על אריחי הרחפן של Uziel ושותפים.

קבוצות (prepare_real_walls.py): dev = 14 אריחים, test = 40 אריחים. החלוקה לפי קבוצות חפיפה.
dev משמש לבחירת הגדרות ולכוונון עדין. אף הגדרה לא נבחרת לפי test.

מודלים:
  all_wall        טריוויאלי: הכול קיר. ה-IoU שלו הוא שיעור הקיר בנתונים.
  bright_texture  כלל ידני: אבן בהירה עם מרקם מקומי (הספים נבחרים על dev).
  segformer_ade   SegFormer מאומן מראש על ADE20K, בלי התאמה. המחלקות wall/building/... = קיר.
  grounded_sam    GroundingDINO + SAM, zero-shot. סף התיבה, סוג הפרומפט (חמש מחלקות או קיר בלבד) וגודל תיבה מרבי נבחרים על dev.
  unet_synth      U-Net שאומן רק על 480 התמונות הסינתטיות (3 seeds). גודל הקלט נבחר על dev.
  unet_real       אותה רשת, ImageNet בלבד, כוונון על 14 אריחי dev (3 seeds).
  unet_synth_real אותה רשת, אתחול מהמודל הסינתטי, אותו כוונון בדיוק על 14 אריחי dev (3 seeds).

unet_real מול unet_synth_real מבודד את תרומת הנתונים הסינתטיים: כל השאר זהה.
הכוונון רץ מספר אפוקים קבוע ומשתמש במודל האחרון, כדי שלא ייבחר checkpoint לפי שום קבוצה.

מדדים: IoU, precision, recall ו-F1 של מחלקת הקיר, מסוכמים על כל הפיקסלים. רווח סמך 95%: bootstrap על קבוצות
חפיפה (2,000 דגימות), כי אריחים חופפים אינם בלתי תלויים.

הערה על ה-test: ההערכה רצה שלוש פעמים. אחרי הריצה הראשונה והשנייה נמצא שרשתות החיפוש של שני מודלי הבסיס
(bright_texture ו-grounded_sam) נחתכו בקצה, והן הורחבו על dev בלבד. המודלים המאומנים והגדרותיהם לא השתנו.

שימוש:
    python src/real_walls_eval.py --stage finetune      # unet_real ו-unet_synth_real, 3 seeds לכל אחד
    python src/real_walls_eval.py --stage eval          # בחירה על dev ואז test, לכל המודלים
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import IGNORE, NUM_CLASSES, PROJECT, REAL_DIR, ROOT, RUNS  # noqa: E402
from datasets import SegDataset, real_items, to_tensor  # noqa: E402

WALL = 2
WALLS = REAL_DIR / "walls"
OUT = PROJECT / "results" / "metrics"
PRED = PROJECT / "results" / "predictions" / "real_walls"
SEEDS = (0, 1, 2)
FT = {"epochs": 30, "repeat": 10, "batch": 8, "crop": 384, "lr": 3e-4, "enc_lr": 3e-5}


def groups() -> dict[str, int]:
    with open(ROOT / "annotation" / "real_walls_manifest.csv", encoding="utf-8") as f:
        return {r["tile"]: int(r["group"]) for r in csv.DictReader(f)}


def counts(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    v = gt != IGNORE
    g = gt[v] == WALL
    p = pred[v].astype(bool)
    return np.array([(p & g).sum(), (p & ~g).sum(), (~p & g).sum(), (~p & ~g).sum()], dtype=np.int64)  # TP FP FN TN


def scores(c: np.ndarray) -> dict:
    tp, fp, fn, _ = c.astype(float)
    if tp + fp + fn == 0:                    # אין קיר ולא נחזה קיר: ה-IoU לא מוגדר (אריח אחד ב-test)
        return {"iou": float("nan"), "precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
    return {"iou": tp / max(tp + fp + fn, 1), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1)}


def cluster_bootstrap(cs: np.ndarray, grp: np.ndarray, n: int = 2000, seed: int = 0) -> np.ndarray:
    """דגימות bootstrap של IoU, בדגימה עם החזרה של קבוצות חפיפה. אותו seed = אותן דגימות לכל המודלים."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(grp)
    per = np.stack([cs[grp == u].sum(0) for u in uniq])
    idx = rng.integers(0, len(uniq), (n, len(uniq)))
    s = per[idx].sum(1).astype(float)
    return s[:, 0] / np.maximum(s[:, 0] + s[:, 1] + s[:, 2], 1)


# ------------------------------------------------------------------ מודלים

class AllWall:
    def __call__(self, im):
        return np.ones((im.height, im.width), bool)


class BrightTexture:
    """אבן בהירה יחסית לסביבה, עם מרקם מקומי. שני הספים נבחרים על dev."""

    def __init__(self, bright: float, texture: float):
        self.b, self.t = bright, texture

    def __call__(self, im):
        g = np.asarray(im.convert("L"), dtype=np.float32) / 255
        loc = ndimage.uniform_filter(g, 41)
        std = np.sqrt(np.maximum(ndimage.uniform_filter(g * g, 15) - ndimage.uniform_filter(g, 15) ** 2, 0))
        m = (ndimage.uniform_filter(g, 9) - ndimage.uniform_filter(g, 151) > self.b) & (std > self.t) & (loc > 0.15)
        m = ndimage.binary_opening(ndimage.binary_closing(m, iterations=4), iterations=3)
        lab, n = ndimage.label(m)
        if n:
            size = ndimage.sum(m, lab, np.arange(1, n + 1))
            m = np.isin(lab, 1 + np.flatnonzero(size > 600))
        return m


class SegFormerADE:
    def __init__(self):
        from hallucination import SegformerEvaluator
        from mapping import load_mapping
        self.ev = SegformerEvaluator(load_mapping())

    def __call__(self, im):
        return self.ev(np.array(im.convert("RGB"))) == WALL


WALL_ONLY = {"wall": ["stone wall", "wall foundation", "ruined wall", "row of stones"]}


class GSAM:
    """prompts: "five" = הפרומפטים של כל חמש המחלקות (כמו ב-test הסינתטי), "wall" = ביטויי קיר בלבד.
    בהערכת קירות, פרומפט של קירות בלבד הוא מודל הבסיס החזק יותר, ולכן שניהם מועמדים על dev."""

    def __init__(self, thr: float, prompts: str = "five", max_box_frac: float = 1.0, model=None):
        from zeroshot import DEFAULT_PROMPTS, GroundedSAM
        self.m = model or GroundedSAM()
        self.m.box_threshold = thr
        self.m.text_threshold = min(thr, 0.25)      # בלי זה, סף טקסט קבוע של 0.25 מבטל כל סף תיבה נמוך ממנו
        self.m.max_box_frac = max_box_frac
        self.m.prompts = WALL_ONLY if prompts == "wall" else DEFAULT_PROMPTS

    def __call__(self, im):
        return self.m(im) == WALL


class UNet:
    def __init__(self, ckpt: Path, short_side: int):
        from train_unet import predict
        from unet import ResNet18UNet
        self.predict, self.s = predict, short_side
        self.model = ResNet18UNet(NUM_CLASSES, pretrained=False).cuda().eval()
        self.model.load_state_dict(torch.load(ckpt, map_location="cuda", weights_only=True)["model"])

    def __call__(self, im):
        k = self.s / min(im.size)
        x = im.resize((round(im.width * k), round(im.height * k)), Image.BILINEAR)
        p = self.predict(self.model, to_tensor(np.array(x))[None].cuda())[0].cpu().numpy().astype(np.uint8)
        return np.array(Image.fromarray(p).resize(im.size, Image.NEAREST)) == WALL


# ------------------------------------------------------------------ שלבים

def run(model, items: list[dict], save: Path | None = None) -> np.ndarray:
    out = []
    for it in items:
        im = Image.open(it["image_path"]).convert("RGB")
        gt = np.array(Image.open(it["label_path"]))
        p = model(im)
        if save:
            save.mkdir(parents=True, exist_ok=True)
            Image.fromarray((p * 255).astype(np.uint8)).save(save / f"{it['id']}.png", optimize=True)
        out.append(counts(p, gt))
    return np.stack(out)


def finetune() -> None:
    from train_unet import class_weights, soft_dice
    from unet import ResNet18UNet
    import torch.nn.functional as F
    base = real_items(WALLS / "dev")
    for arm, use_synth in (("unet_real", False), ("unet_synth_real", True)):
        for seed in SEEDS:
            out = RUNS / f"{arm}_s{seed}"
            if (out / "last.pt").exists():
                print(f"{out.name}: קיים")
                continue
            torch.manual_seed(seed)
            np.random.seed(seed)
            ds = SegDataset(base * FT["repeat"], train=True, crop=FT["crop"], boundary_px=2, seed=seed)
            dl = torch.utils.data.DataLoader(ds, batch_size=FT["batch"], shuffle=True, drop_last=True)
            model = ResNet18UNet(NUM_CLASSES, pretrained=True).cuda()
            if use_synth:
                st = torch.load(RUNS / f"base_s{seed}" / "best.pt", map_location="cuda", weights_only=True)
                model.load_state_dict(st["model"])
            w = class_weights(base).cuda()
            opt = torch.optim.AdamW([{"params": model.encoder_parameters(), "lr": FT["enc_lr"]},
                                     {"params": model.decoder_parameters(), "lr": FT["lr"]}], weight_decay=1e-4)
            steps = FT["epochs"] * len(dl)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
            scaler, t0, log = torch.amp.GradScaler("cuda"), time.perf_counter(), []
            for ep in range(FT["epochs"]):
                tot = 0.0
                for x, y, _ in dl:
                    x, y = x.cuda(), y.cuda()
                    with torch.autocast("cuda", dtype=torch.float16):
                        lg = model(x)
                        loss = F.cross_entropy(lg.float(), y, weight=w, ignore_index=IGNORE) + 0.5 * soft_dice(lg, y)
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                    sched.step()
                    tot += loss.item()
                log.append(round(tot / len(dl), 4))
            out.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict()}, out / "last.pt")
            (out / "config.json").write_text(json.dumps({**FT, "seed": seed, "arm": arm, "n_tiles": len(base),
                                                         "init": f"base_s{seed}" if use_synth else "ImageNet",
                                                         "seconds": round(time.perf_counter() - t0), "loss": log}, indent=1))
            print(f"{out.name}: loss {log[0]} -> {log[-1]}, {time.perf_counter() - t0:.0f}s", flush=True)
            del model
            torch.cuda.empty_cache()


def evaluate(reuse: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    grp = groups()
    dev, test = real_items(WALLS / "dev"), real_items(WALLS / "test")
    res: dict = {"n_dev": len(dev), "n_test": len(test), "n_test_groups": len({grp[i["id"]] for i in test}), "selected_on_dev": {}}

    prev = json.loads((OUT / "real_walls.json").read_text())["selected_on_dev"] if reuse else {}

    def choose(name, variants):
        if reuse:                            # ההגדרות כבר נבחרו על dev בריצה קודמת; לא בוחרים שוב
            res["selected_on_dev"][name] = prev[name]
            return next(k for k in variants if str(k) == prev[name]["choice"])
        dev_iou = {k: scores(run(mk(), dev).sum(0))["iou"] for k, mk in variants.items()}
        key = max(dev_iou, key=dev_iou.get)
        res["selected_on_dev"][name] = {"choice": str(key), "dev_iou": round(dev_iou[key], 4),
                                        "all": {str(k): round(v, 4) for k, v in dev_iou.items()}}
        print(name, res["selected_on_dev"][name], flush=True)
        return key

    models: dict[str, list] = {"all_wall": [AllWall]}
    bt = choose("bright_texture", {(b, t): (lambda b=b, t=t: BrightTexture(b, t))
                                   for b in (-0.02, 0.0, 0.02, 0.04, 0.06, 0.1, 0.14, 0.18) for t in (0.11, 0.14, 0.17, 0.2, 0.23, 0.26)})
    models["bright_texture"] = [lambda: BrightTexture(*bt)]
    models["segformer_ade"] = [SegFormerADE]
    from zeroshot import GroundedSAM
    gs = GroundedSAM()
    pr, thr, fr = choose("grounded_sam", {(q, t, f): (lambda q=q, t=t, f=f: GSAM(t, q, f, gs))
                                          for q in ("five", "wall") for t in (0.08, 0.12, 0.16, 0.2, 0.25, 0.3) for f in (1.0, 0.6, 0.4, 0.25, 0.15)})
    models["grounded_sam"] = [lambda: GSAM(thr, pr, fr, gs)]
    side = choose("unet_synth", {s: (lambda s=s: UNet(RUNS / "base_s0" / "best.pt", s)) for s in (384, 512, 800, 1136)})
    models["unet_synth"] = [lambda sd=sd: UNet(RUNS / f"base_s{sd}" / "best.pt", side) for sd in SEEDS]
    models["unet_real"] = [lambda sd=sd: UNet(RUNS / f"unet_real_s{sd}" / "last.pt", 800) for sd in SEEDS]
    models["unet_synth_real"] = [lambda sd=sd: UNet(RUNS / f"unet_synth_real_s{sd}" / "last.pt", 800) for sd in SEEDS]

    g = np.array([grp[i["id"]] for i in test])
    per_tile, boots, res["models"] = [], {}, {}
    for name, mks in models.items():
        runs = [run(mk(), test, PRED / name if k == 0 else None) for k, mk in enumerate(mks)]
        pts = [scores(r.sum(0)) for r in runs]
        b = np.mean([cluster_bootstrap(r, g) for r in runs], axis=0)      # ממוצע על seeds בתוך כל דגימה
        boots[name] = b
        res["models"][name] = {
            **{k: round(float(np.mean([p[k] for p in pts])), 4) for k in pts[0]},
            "iou_ci95": [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)],
            "iou_per_seed": [round(p["iou"], 4) for p in pts], "n_seeds": len(pts)}
        for it, c in zip(test, np.mean(runs, axis=0)):
            per_tile.append({"tile": it["id"], "group": grp[it["id"]], "model": name,
                             **{k: round(v, 4) for k, v in scores(c).items()}})
        print(name, res["models"][name], flush=True)

    def delta(a, b):
        d = boots[a] - boots[b]
        return {"delta_iou": round(res["models"][a]["iou"] - res["models"][b]["iou"], 4),
                "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]}

    res["paired"] = {"unet_real - bright_texture": delta("unet_real", "bright_texture"),
                     "grounded_sam - all_wall": delta("grounded_sam", "all_wall"),
                     "unet_synth - grounded_sam": delta("unet_synth", "grounded_sam"),
                     "unet_synth_real - unet_real": delta("unet_synth_real", "unet_real"),
                     "unet_real - grounded_sam": delta("unet_real", "grounded_sam"),
                     "unet_synth - segformer_ade": delta("unet_synth", "segformer_ade")}
    (OUT / "real_walls.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    with open(OUT / "real_walls_per_tile.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_tile[0]))
        w.writeheader()
        w.writerows(per_tile)
    print(json.dumps(res["paired"], indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["finetune", "eval"], required=True)
    ap.add_argument("--reuse-selection", action="store_true", help="הגדרות dev מתוך real_walls.json הקיים, בלי חיפוש חוזר")
    a = ap.parse_args()
    finetune() if a.stage == "finetune" else evaluate(a.reuse_selection)
