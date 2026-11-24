"""הכנת קבוצה אמיתית של קירות מתוך Uziel, Zohar & Jaffe 2026 (Zenodo 10.5281/zenodo.15673668, MIT).

הקלט: 56 אריחי רחפן של 800x800 מאתרים בישראל, עם פוליגונים של קטעי קיר בפורמט LabelMe.
הפלט: data/real/walls/{dev,test}/{images,masks} ו-manifest שנשמר בגיט (annotation/real_walls_manifest.csv).

מה נעשה, ולמה:
  - שמות התוויות במקור לא עקביים ('..', '\\\\\\', 'stone', '.'). כולם מסמנים קטעי קיר, ולכן כל פוליגון = wall.
  - אזורים שחורים (מחוץ לאורתופוטו) מקבלים IGNORE.
  - אריחים שכנים חופפים ב-25%, ושני זוגות זהים לגמרי ותויגו פעמיים בשמות שונים. החפיפות מזוהות לפי phase
    correlation, והחלוקה ל-dev ול-test היא לפי קבוצת חפיפה, כדי ששטח אחד לא יופיע בשני הצדדים.
  - מכל זוג זהה נשאר אריח אחד. התיוג השני נשמר בצד, ו-IoU בין שני התיוגים נותן אומדן ראשון לעקביות התיוג.
  - מקור הנתונים לא מציין מאיזה אתר כל אריח, ולכן אי אפשר לחלק לפי אתר. זו מגבלה, והיא מדווחת.
  - רק הקירות מתויגים. פיקסל שאינו קיר יכול להיות רצפה, בור או אדמה, ולכן ההערכה היא בינארית: קיר או לא.

שימוש:
    python src/prepare_real_walls.py
"""

from __future__ import annotations

import csv
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import IGNORE, REAL_DIR, ROOT  # noqa: E402

SRC = REAL_DIR / "uziel_extracted" / "annotated_tiles" / "wall_tiles"
OUT = REAL_DIR / "walls"
MANIFEST = ROOT / "annotation" / "real_walls_manifest.csv"
WALL = 2
DEV_SHARE = 0.25
OVERLAP_THR = 0.028      # בין הזוג החופף החלש ביותר (0.030) לזוג הלא-חופף החזק ביותר (0.027). אומת בנפרד ב-NCC מלא
DUPLICATE_THR = 0.95


def nodata_mask(img: np.ndarray, min_px: int = 2000) -> np.ndarray:
    """אזור שחור רציף וגדול. צל או אבן כהה בודדים לא נחשבים."""
    dark = img.max(axis=2) <= 3
    lab, n = ndimage.label(dark)
    if not n:
        return np.zeros(dark.shape, bool)
    sizes = ndimage.sum(dark, lab, index=np.arange(1, n + 1))
    return np.isin(lab, 1 + np.flatnonzero(sizes >= min_px))


def wall_mask(shapes: list[dict], size: tuple[int, int]) -> np.ndarray:
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    for s in shapes:
        if len(s["points"]) >= 3:
            d.polygon([tuple(p) for p in s["points"]], fill=1)
    return np.array(m).astype(bool)


def _norm(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    return (a - a.mean()) / (a.std() + 1e-6)


def overlap_score(a: np.ndarray, b: np.ndarray) -> tuple[float, tuple[int, int]]:
    """שיא הקורלציה הצולבת (phase correlation) בין שתי תמונות אפורות קטנות, וההזחה שבה הוא התקבל.
    ערך גבוה אומר ששתי התמונות מצלמות אותו שטח, גם אם הן מוזחות זו ביחס לזו."""
    F = np.fft.fft2(_norm(a)) * np.conj(np.fft.fft2(_norm(b)))
    r = np.fft.ifft2(F / (np.abs(F) + 1e-6)).real
    i = int(r.argmax())
    dy, dx = divmod(i, r.shape[1])
    return float(r.max()), (dy, dx)


def group_overlaps(gray: dict[str, np.ndarray], thr: float) -> tuple[dict[str, int], list[tuple]]:
    names = sorted(gray)
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            s, shift = overlap_score(gray[a], gray[b])
            if s >= thr:
                pairs.append((a, b, round(s, 3), shift))
                parent[find(b)] = find(a)
    roots = sorted({find(n) for n in names})
    return {n: roots.index(find(n)) for n in names}, pairs


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    files = sorted(SRC.glob("*.json"))
    if not files:
        raise SystemExit(f"לא נמצאו אריחים ב-{SRC}. מורידים את StonesTracker-project.zip ומחלצים את annotated_tiles.")
    data, gray = {}, {}
    for f in files:
        img = Image.open(f.with_suffix(".jpg")).convert("RGB")
        arr = np.array(img)
        j = json.loads(f.read_text(encoding="utf-8"))
        wall, nod = wall_mask(j["shapes"], img.size), nodata_mask(arr)
        lab = np.zeros(wall.shape, np.uint8)
        lab[wall] = WALL
        lab[nod] = IGNORE
        data[f.stem] = {"img": img, "lab": lab, "n_polygons": len(j["shapes"]),
                        "labels": sorted({s["label"] for s in j["shapes"]}),
                        "nodata": float(nod.mean())}
        gray[f.stem] = np.array(img.convert("L").resize((200, 200), Image.BILINEAR))
    groups, pairs = group_overlaps(gray, thr=OVERLAP_THR)
    duplicates = {b: a for a, b, sc, _ in pairs if sc >= DUPLICATE_THR}      # b זהה ל-a, תויג פעם שנייה
    print(f"{len(files)} אריחים, {len(set(groups.values()))} קבוצות אחרי איחוד אריחים חופפים")
    for p in pairs:
        print("  חופפים:", p)

    rng = random.Random(0)
    members: dict[int, list[str]] = {}
    for nm in data:
        if nm not in duplicates:
            members.setdefault(groups[nm], []).append(nm)
    gids = sorted(members)
    rng.shuffle(gids)
    total, dev_groups, n_dev = sum(map(len, members.values())), set(), 0
    for g in gids:                               # ממלאים את dev בקבוצות שלמות, עד כרבע מהאריחים
        if n_dev + len(members[g]) <= round(total * DEV_SHARE):
            dev_groups.add(g)
            n_dev += len(members[g])
    agreement = []
    for b, a in duplicates.items():
        wa, wb = data[a]["lab"] == WALL, data[b]["lab"] == WALL
        agreement.append({"a": a, "b": b, "iou": round(float((wa & wb).sum() / max(1, (wa | wb).sum())), 4)})
    print("עקביות תיוג על אריחים זהים:", agreement)
    if OUT.exists():
        shutil.rmtree(OUT)
    rows = []
    for name, d in data.items():
        split = "duplicate" if name in duplicates else "dev" if groups[name] in dev_groups else "test"
        for sub in ("images", "masks"):
            (OUT / split / sub).mkdir(parents=True, exist_ok=True)
        d["img"].save(OUT / split / "images" / f"{name}.png")
        Image.fromarray(d["lab"]).save(OUT / split / "masks" / f"{name}.png")
        valid = d["lab"] != IGNORE
        rows.append({"tile": name, "split": split, "group": groups[name], "n_polygons": d["n_polygons"],
                     "wall_frac": round(float((d["lab"][valid] == WALL).mean()), 4),
                     "nodata_frac": round(d["nodata"], 4), "source_labels": "|".join(d["labels"])})
    MANIFEST.parent.mkdir(exist_ok=True)
    with open(MANIFEST, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (MANIFEST.parent / "real_walls_overlaps.json").write_text(
        json.dumps({"annotation_agreement_on_identical_tiles": agreement,
                    "overlaps": [{"a": a, "b": b, "score": s, "shift_px_of_200": list(sh)} for a, b, s, sh in pairs]},
                   indent=1),
        encoding="utf-8")
    for sp in ("dev", "test"):
        r = [x for x in rows if x["split"] == sp]
        print(f"{sp}: {len(r)} אריחים, {len({x['group'] for x in r})} קבוצות, "
              f"שיעור קיר ממוצע {np.mean([x['wall_frac'] for x in r]):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
