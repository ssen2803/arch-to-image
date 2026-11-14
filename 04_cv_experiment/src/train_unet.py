"""אימון ה-U-Net על הנתונים הסינתטיים של שלב 3. מתאים ל-6GB VRAM (FP16, batch קטן).

Loss = CrossEntropy (עם משקלי מחלקות 1/sqrt(שכיחות), כי מתקנים הם כ-2% מהפיקסלים)
       + 0.5 * Soft Dice. פיקסלים עם IGNORE לא נספרים.
אופטימייזר: AdamW, קצב למידה נמוך פי 10 למקודד המאומן מראש, cosine עם חימום של epoch אחד.
בחירת checkpoint: לפי mIoU על ולידציה סינתטית (מפוצלת לפי Layout). אם יש real-dev, הוא
מדווח לצורך מעקב בלבד; ראו README לגבי הוגנות מול ה-zero-shot.

סינון זליגת תוויות (מקרה קצה 1), שתי אפשרויות:
  --min-agree / --max-har          לפי מדדי SegFormer שנשמרו ב-manifest (מעריך חיצוני)
  --filter-csv + --drop-frac       לפי ציון cross-fit (crossfit_filter.py): מוציאים את החלק
                                   היחסי של התמונות שהתיוג שלהן הכי לא מתיישב עם חיזוי out-of-fold

שימוש:
    python src/train_unet.py --name base
    python src/train_unet.py --name crossfit20 --filter-csv runs/crossfit/scores.csv --drop-frac 0.2
    python src/train_unet.py --name ood_pit_in_floor --exclude-tags pit_in_floor
    python src/train_unet.py --name ood_dir --exclude-tags wall_south_of_pit --no-geometric-aug
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import IGNORE, NUM_CLASSES, REAL_DIR, RUNS, SYNTH_DIR  # noqa: E402
from compositions import composition_tags  # noqa: E402
from datasets import SegDataset, filter_rows, read_manifest, real_items, split_by_layout  # noqa: E402
from metrics import confusion, iou_from_confusion, miou  # noqa: E402
from unet import ResNet18UNet, count_parameters  # noqa: E402

DEFAULT_HP = {"epochs": 40, "batch": 8, "crop": 384, "lr": 1e-3, "enc_lr": 1e-4, "boundary_ignore": 3,
              "geometric_aug": True, "seed": 0}


def class_weights(items: list[dict]) -> torch.Tensor:
    counts = np.zeros(NUM_CLASSES)
    seen = set()
    for it in items:
        if it["label_path"] in seen:
            continue
        seen.add(it["label_path"])
        lab = np.array(Image.open(it["label_path"]))
        counts += np.bincount(lab[lab != IGNORE].ravel(), minlength=NUM_CLASSES)[:NUM_CLASSES]
    freq = counts / counts.sum()
    w = 1.0 / np.sqrt(np.maximum(freq, 1e-4))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def soft_dice(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = (target != IGNORE).unsqueeze(1)
    probs = logits.float().softmax(1) * valid
    onehot = F.one_hot(target.clamp(0, NUM_CLASSES - 1), NUM_CLASSES).permute(0, 3, 1, 2) * valid
    inter = (probs * onehot).sum((0, 2, 3))
    denom = probs.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    present = onehot.sum((0, 2, 3)) > 0
    dice = (2 * inter + 1) / (denom + 1)
    return 1 - dice[present].mean() if present.any() else logits.sum() * 0


@torch.no_grad()
def predict(model, x: torch.Tensor) -> torch.Tensor:
    """ריפוד לכפולה של 32, חיזוי, וחיתוך חזרה."""
    h, w = x.shape[-2:]
    ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32
    xp = F.pad(x, (0, pw, 0, ph), mode="reflect")
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(xp)
    return logits[..., :h, :w].argmax(1)


@torch.no_grad()
def evaluate_loader(model, loader, device) -> dict:
    model.eval()
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES + 1), dtype=np.int64)
    for x, y, _ in loader:
        pred = predict(model, x.to(device)).cpu().numpy()
        for p, t in zip(pred, y.numpy()):
            cm += confusion(p, t)
    model.train()
    return {"miou": round(miou(cm), 4), "iou": [round(float(v), 4) for v in iou_from_confusion(cm)]}


def fit(train_rows: list[dict], val_rows: list[dict], hp: dict, out: Path | None = None,
        dev_dl=None, extra_config: dict | None = None):
    """לולאת האימון המשותפת (train_unet ו-crossfit_filter). מחזיר (מודל, לוג, mIoU הטוב ביותר)."""
    hp = {**DEFAULT_HP, **hp}
    torch.manual_seed(hp["seed"])
    device = "cuda"
    train_ds = SegDataset(train_rows, train=True, crop=hp["crop"], geometric_aug=hp["geometric_aug"],
                          boundary_px=hp["boundary_ignore"], seed=hp["seed"])
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=hp["batch"], shuffle=True, num_workers=0,
                                           drop_last=True)
    val_dl = torch.utils.data.DataLoader(SegDataset(val_rows, train=False), batch_size=4, num_workers=0)

    model = ResNet18UNet(NUM_CLASSES, pretrained=True).to(device)
    weights = class_weights(train_rows).to(device)
    opt = torch.optim.AdamW([{"params": model.encoder_parameters(), "lr": hp["enc_lr"]},
                             {"params": model.decoder_parameters(), "lr": hp["lr"]}], weight_decay=1e-4)
    steps, warm = hp["epochs"] * len(train_dl), len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda")

    config = {**hp, **(extra_config or {}), "n_train": len(train_rows), "n_val": len(val_rows),
              "class_weights": [round(float(w), 3) for w in weights], "params": count_parameters(model)}
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    log, best, best_state = [], -1.0, None
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, hp["epochs"] + 1):
        t0, total = time.perf_counter(), 0.0
        for x, y, _ in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
                loss = F.cross_entropy(logits.float(), y, weight=weights, ignore_index=IGNORE) \
                    + 0.5 * soft_dice(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        val = evaluate_loader(model, val_dl, device)
        entry = {"epoch": epoch, "loss": round(total / len(train_dl), 4), "val_miou": val["miou"],
                 "val_iou": val["iou"], "seconds": round(time.perf_counter() - t0, 1),
                 "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20)}
        if dev_dl is not None:
            entry["real_dev_miou"] = evaluate_loader(model, dev_dl, device)["miou"]
        log.append(entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)
        if val["miou"] > best:
            best = val["miou"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if out:
                torch.save({"model": best_state, "epoch": epoch, "val_miou": best, "config": config}, out / "best.pt")
        if out:
            (out / "log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    model.load_state_dict(best_state)
    return model, log, best


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", type=Path, default=SYNTH_DIR)
    ap.add_argument("--epochs", type=int, default=DEFAULT_HP["epochs"])
    ap.add_argument("--batch", type=int, default=DEFAULT_HP["batch"])
    ap.add_argument("--crop", type=int, default=DEFAULT_HP["crop"])
    ap.add_argument("--lr", type=float, default=DEFAULT_HP["lr"], help="מפענח")
    ap.add_argument("--enc-lr", type=float, default=DEFAULT_HP["enc_lr"], help="מקודד מאומן מראש")
    ap.add_argument("--min-agree", type=float, default=0.0, help="סינון SegFormer: agree_miou מינימלי")
    ap.add_argument("--max-har", type=float, default=1.0, help="סינון SegFormer: HAR מקסימלי")
    ap.add_argument("--filter-csv", type=Path, help="ציוני cross-fit (crossfit_filter.py)")
    ap.add_argument("--drop-frac", type=float, default=0.0, help="איזה חלק מהתמונות עם הציון הנמוך להוציא")
    ap.add_argument("--boundary-ignore", type=int, default=DEFAULT_HP["boundary_ignore"])
    ap.add_argument("--no-geometric-aug", action="store_true", help="חובה בניסוי קומפוזיציות כיווניות")
    ap.add_argument("--exclude-tags", nargs="*", default=[], help="קומפוזיציות שיוצאו מהאימון (ניסוי OOD)")
    ap.add_argument("--real-dev", type=Path, default=REAL_DIR / "dev")
    ap.add_argument("--max-train", type=int, default=0, help="להרצת עשן בלבד")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = read_manifest(args.data)
    train_rows, val_rows = split_by_layout(rows)
    n_before = len(train_rows)
    train_rows = filter_rows(train_rows, args.min_agree, args.max_har)
    if args.filter_csv and args.drop_frac > 0:
        with open(args.filter_csv, encoding="utf-8") as f:
            score = {r["image"]: float(r["oof_miou"]) for r in csv.DictReader(f)}
        ranked = sorted(train_rows, key=lambda r: score.get(r["image"], 1.0))
        train_rows = ranked[int(len(ranked) * args.drop_frac):]
    held_out = []
    if args.exclude_tags:
        keep = []
        for r in train_rows:
            tags = composition_tags(np.array(Image.open(r["label_path"])))
            (held_out if tags & set(args.exclude_tags) else keep).append(r)
        train_rows = keep
    if args.max_train:
        train_rows = train_rows[:args.max_train]
    print(f"train: {len(train_rows)} (מתוך {n_before} לפני סינון/הוצאה), val: {len(val_rows)}, "
          f"הוצאו בגלל קומפוזיציה: {len(held_out)}")

    dev_dl = None
    if args.real_dev.exists() and any((args.real_dev / "images").glob("*")):
        dev_dl = torch.utils.data.DataLoader(SegDataset(real_items(args.real_dev), train=False), batch_size=1)
    out = RUNS / args.name
    hp = {"epochs": args.epochs, "batch": args.batch, "crop": args.crop, "lr": args.lr, "enc_lr": args.enc_lr,
          "boundary_ignore": args.boundary_ignore, "geometric_aug": not args.no_geometric_aug, "seed": args.seed}
    extra = {"filters": {"min_agree": args.min_agree, "max_har": args.max_har,
                         "filter_csv": str(args.filter_csv) if args.filter_csv else None, "drop_frac": args.drop_frac},
             "exclude_tags": args.exclude_tags, "n_held_out": len(held_out)}
    if held_out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "held_out_images.txt").write_text("\n".join(str(r["image_path"]) for r in held_out), encoding="utf-8")
    _, _, best = fit(train_rows, val_rows, hp, out, dev_dl, extra)
    print(f"best synthetic val mIoU = {best:.4f} -> {out / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
