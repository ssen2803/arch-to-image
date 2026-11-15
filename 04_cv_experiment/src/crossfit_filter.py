"""מסנן זליגת תוויות מבוסס cross-fit (מקרה קצה 1), בלי מעריך חיצוני.

הבעיה: ControlNet לא תמיד מצייר את מה שבמסכה, ואז המסכה היא תיוג שגוי של התמונה.
מעריך מוכן (SegFormer, Grounding DINO) סובל מפער תחום על התמונות האלה, ובנוסף
Grounding DINO הוא חלק מה-baseline של ה-zero-shot. סינון לפיו היה מטה את ההשוואה
לטובת תמונות ש-zero-shot כבר "מבין".

הפתרון: מחלקים את ה-Layouts ל-k קבוצות. מאמנים U-Net על k-1 קבוצות ומנבאים על
הקבוצה שנשארה בחוץ (out-of-fold). תמונה שהתיוג שלה לא מתיישב עם חיזוי של מודל
שלא ראה אותה, אבל אומן על אותו תחום, היא חשודה בזליגה. זו שיטה מקובלת לזיהוי
תוויות רועשות (כמו confident learning). מדד הציון: mIoU בין החיזוי לתיוג ברזולוציה הגסה.

שימוש:
    python src/crossfit_filter.py --folds 2 --epochs 15
    python src/train_unet.py --name crossfit20 --filter-csv runs/crossfit/scores.csv --drop-frac 0.2
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUNS, SYNTH_DIR  # noqa: E402
from datasets import SegDataset, drop_blank_images, read_manifest, split_by_layout  # noqa: E402
from metrics import coarsen_labels, confusion, miou  # noqa: E402
from train_unet import fit, predict  # noqa: E402


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=SYNTH_DIR)
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", type=Path, default=RUNS / "crossfit")
    args = ap.parse_args()

    rows, blank = drop_blank_images(read_manifest(args.data))
    if blank:
        print(f"הוצאו {len(blank)} תמונות ריקות: {', '.join(Path(r['image']).name for r in blank)}")
    train_rows, val_rows = split_by_layout(rows)   # הוולידציה הסינתטית לא משתתפת בסינון
    layouts = sorted({r["layout_id"] for r in train_rows})
    fold_of = {lid: i % args.folds for i, lid in enumerate(layouts)}
    args.out.mkdir(parents=True, exist_ok=True)
    scores = []
    for k in range(args.folds):
        fit_rows = [r for r in train_rows if fold_of[r["layout_id"]] != k]
        oof_rows = [r for r in train_rows if fold_of[r["layout_id"]] == k]
        print(f"fold {k}: אימון על {len(fit_rows)}, חיזוי על {len(oof_rows)}", flush=True)
        model, _, _ = fit(fit_rows, val_rows, {"epochs": args.epochs, "batch": args.batch, "seed": k})
        model.eval()
        for x, y, path in torch.utils.data.DataLoader(SegDataset(oof_rows, train=False), batch_size=4):
            pred = predict(model, x.cuda()).cpu().numpy()
            for p, t, ip in zip(pred, y.numpy(), path):
                cm = confusion(coarsen_labels(p.astype(np.uint8)), coarsen_labels(t.astype(np.uint8)), n=3)
                image = str(Path(ip).relative_to(args.data)).replace("\\", "/")
                scores.append({"image": image, "fold": k, "oof_miou": round(miou(cm), 4)})
        del model
        torch.cuda.empty_cache()
    with open(args.out / "scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "fold", "oof_miou"])
        w.writeheader()
        w.writerows(scores)
    v = sorted(s["oof_miou"] for s in scores)
    print(f"{len(scores)} ציונים -> {args.out / 'scores.csv'} | רבעונים: "
          f"{v[len(v) // 4]:.3f} / {v[len(v) // 2]:.3f} / {v[3 * len(v) // 4]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
