"""מסנן זליגת תוויות מבוסס cross-fit (מקרה קצה 1), בלי מעריך חיצוני.

הבעיה: ControlNet לא תמיד מצייר את מה שבמסכה, ואז המסכה היא תיוג שגוי של התמונה.
מעריך מוכן (SegFormer, Grounding DINO) סובל מפער תחום על התמונות האלה, ובנוסף
Grounding DINO הוא חלק מה-baseline של ה-zero-shot. סינון לפיו היה מטה את ההשוואה
לטובת תמונות ש-zero-shot כבר "מבין".

הפתרון: מחלקים את ה-Layouts ל-k קבוצות. מאמנים U-Net על k-1 קבוצות ומנבאים על
הקבוצה שנשארה בחוץ (out-of-fold). תמונה שהתיוג שלה לא מתיישב עם חיזוי של מודל
שלא ראה אותה, אבל אומן על אותו תחום, היא חשודה בזליגה. זו שיטה מקובלת לזיהוי
תוויות רועשות (כמו confident learning). מדד הציון: mIoU בין החיזוי לתיוג ברזולוציה הגסה.

יציבות: הדירוג רועש. בין שתי ריצות זהות נמדד Spearman של 0.88, ורק 77% מה-10% התחתונים
היו משותפים (README, סעיף 5.1). לכן --repeats מריץ את כל התהליך כמה פעמים, עם חלוקה אחרת
לקיפולים ו-seed אחר בכל חזרה, והציון הסופי הוא הממוצע. עמודת oof_std מראה כמה ציון משתנה.

שימוש:
    python src/crossfit_filter.py --folds 2 --epochs 15
    python src/crossfit_filter.py --folds 2 --epochs 15 --repeats 3
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
    ap.add_argument("--repeats", type=int, default=1, help="חזרות עם חלוקה ו-seed שונים; הציון הוא הממוצע")
    args = ap.parse_args()

    rows, blank = drop_blank_images(read_manifest(args.data))
    if blank:
        print(f"הוצאו {len(blank)} תמונות ריקות: {', '.join(Path(r['image']).name for r in blank)}")
    train_rows, val_rows = split_by_layout(rows)   # הוולידציה הסינתטית לא משתתפת בסינון
    layouts = sorted({r["layout_id"] for r in train_rows})
    args.out.mkdir(parents=True, exist_ok=True)
    per_repeat: list[dict[str, float]] = []
    fold0: dict[str, int] = {}
    for rep in range(args.repeats):
        order = list(layouts)
        if rep > 0:                                  # חזרה 0 שומרת על החלוקה המקורית, לשחזור תוצאות קודמות
            np.random.default_rng(rep).shuffle(order)
        fold_of = {lid: i % args.folds for i, lid in enumerate(order)}
        scores: dict[str, float] = {}
        for k in range(args.folds):
            fit_rows = [r for r in train_rows if fold_of[r["layout_id"]] != k]
            oof_rows = [r for r in train_rows if fold_of[r["layout_id"]] == k]
            print(f"repeat {rep} fold {k}: אימון על {len(fit_rows)}, חיזוי על {len(oof_rows)}", flush=True)
            model, _, _ = fit(fit_rows, val_rows, {"epochs": args.epochs, "batch": args.batch, "seed": 100 * rep + k})
            model.eval()
            for x, y, path in torch.utils.data.DataLoader(SegDataset(oof_rows, train=False), batch_size=4):
                pred = predict(model, x.cuda()).cpu().numpy()
                for p, t_, ip in zip(pred, y.numpy(), path):
                    cm = confusion(coarsen_labels(p.astype(np.uint8)), coarsen_labels(t_.astype(np.uint8)), n=3)
                    image = str(Path(ip).relative_to(args.data)).replace("\\", "/")
                    scores[image] = round(miou(cm), 4)
                    if rep == 0:
                        fold0[image] = k
            del model
            torch.cuda.empty_cache()
        per_repeat.append(scores)

    images = sorted(per_repeat[0])
    fields = ["image", "fold", "oof_miou", "oof_std"] + [f"oof_r{r}" for r in range(args.repeats)]
    rows_out = []
    for im in images:
        vals = [s[im] for s in per_repeat]
        rows_out.append({"image": im, "fold": fold0[im], "oof_miou": round(float(np.mean(vals)), 4),
                         "oof_std": round(float(np.std(vals)), 4),
                         **{f"oof_r{r}": v for r, v in enumerate(vals)}})
    with open(args.out / "scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)
    v = sorted(r["oof_miou"] for r in rows_out)
    print(f"{len(rows_out)} ציונים -> {args.out / 'scores.csv'} | רבעונים: "
          f"{v[len(v) // 4]:.3f} / {v[len(v) // 2]:.3f} / {v[3 * len(v) // 4]:.3f}")
    if args.repeats > 1:
        from scipy.stats import spearmanr
        rhos = [spearmanr([per_repeat[a][i] for i in images], [per_repeat[b][i] for i in images]).statistic
                for a in range(args.repeats) for b in range(a + 1, args.repeats)]
        print(f"יציבות בין חזרות: Spearman ממוצע {np.mean(rhos):.3f} (מינימום {min(rhos):.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
