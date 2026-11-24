"""ניתוח שגיאות לשלושת השלבים: חילוץ, ייצור, סגמנטציה. כל קטגוריה נספרת אוטומטית מקובצי התוצאות.

פלט: results/metrics/error_analysis.json

שימוש:
    python src/error_analysis.py
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import IGNORE, PROJECT, REAL_DIR  # noqa: E402
from compare_layouts import compare  # noqa: E402
from layout_core import image_transform, load_layout  # noqa: E402

GEN = PROJECT / "03_controlled_generation" / "outputs"
EX = PROJECT / "02_layout_pipeline" / "examples"
OUT = PROJECT / "results" / "metrics"


# ------------------------------------------------------------------ ייצור

def generation() -> dict:
    out = {}
    for cond in ("photo", "photo_llm", "photo_textonly", "gligen"):
        files = sorted(GEN.glob(f"*/{cond}_s[0-9].hor.json"))
        if not files:
            continue
        n = wall = foreign = 0
        rec = collections.defaultdict(lambda: [0, 0])
        rec_c = collections.defaultdict(lambda: [0, 0])
        pit_hits, foreign_on = [], 0
        for f in files:
            d = json.loads(f.read_text(encoding="utf-8"))["coarse"]
            fine = json.loads(f.read_text(encoding="utf-8"))["fine"]
            L = load_layout(EX / f.parent.name / "layout.gold.json")
            n += 1
            # קיר מומצא: רכיב "בנוי" ללא עיגון, ששטחו לפחות 1% מהתמונה
            wall += any(h["class"] == "wall" and h["area"] >= 0.01 * 512 * 512 for h in d["hallucinated"])
            foreign += d["foreign_components"] > 0
            missed, missed_c = {m["id"] for m in fine["missed"]}, {m["id"] for m in d["missed"]}
            # רכיב זר שחופף לבור או למתקן בפריסה: לא תוכן נוסף, אלא אובייקט שצויר במראה זר
            to_px, _ = image_transform(L, 512)
            boxes = [(*to_px(e["bbox"]["x"], e["bbox"]["y"] + e["bbox"]["h"]), *to_px(e["bbox"]["x"] + e["bbox"]["w"], e["bbox"]["y"]))
                     for e in L["elements"] if e["class"] in ("pit", "installation")]
            fb = [h["bbox"] for h in d["hallucinated"] if h["class"] == "foreign"]
            foreign_on += any(min(b[2], x1) - max(b[0], x0) > 0 and min(b[3], y1) - max(b[1], y0) > 0 for b in fb for x0, y0, x1, y1 in boxes)
            for e in L["elements"]:
                if e["position"] == "unspecified":
                    continue
                rec[e["class"]][1] += 1
                rec[e["class"]][0] += e["id"] not in missed
                if e["class"] != "floor":           # ברזולוציה הגסה רצפה מתמזגת לקרקע ואינה אובייקט
                    rec_c[e["class"]][1] += 1
                    rec_c[e["class"]][0] += e["id"] not in missed_c
                if e["class"] == "pit" and e["id"] not in missed:
                    pit_hits.append(f"{f.parent.name}/{f.stem.replace('.hor', '')}:{e['id']}")
        out[cond] = {"n_images": n, "images_with_invented_wall": wall, "images_with_foreign_content": foreign,
                     "images_with_foreign_on_pit_or_installation": foreign_on,
                     "recall_coarse_by_class": {c: {"drawn": a, "total": b} for c, (a, b) in sorted(rec_c.items())},
                     "recall_fine_by_class": {c: {"drawn": a, "total": b, "recall": round(a / b, 3)} for c, (a, b) in sorted(rec.items())},
                     "pits_counted_as_drawn": pit_hits}
    return out


# ------------------------------------------------------------------ חילוץ

INVERSE = {"south_of": "north_of", "east_of": "west_of"}


def canon(rel: tuple) -> tuple:
    """צורה קנונית של יחס: "A מדרום ל-B" ו-"B מצפון ל-A" הם אותה טענה, ו-adjacent_to סימטרי.
    ההשוואה המחמירה (compare_layouts) סופרת אותם כשונים."""
    s, p, o = rel
    if p in INVERSE:
        return o, INVERSE[p], s
    if p == "adjacent_to":
        return (*sorted((str(s), str(o)))[:1], p, sorted((str(s), str(o)))[1])
    return rel


def extraction(name: str) -> dict | None:
    miss, extra, dist, ious = collections.Counter(), collections.Counter(), collections.defaultdict(list), collections.defaultdict(list)
    n = 0
    strict, can = np.zeros(3, int), np.zeros(3, int)                    # TP, predicted, gold
    for d in sorted(EX.iterdir()):
        p = d / name
        if not p.exists():
            continue
        n += 1
        pred, gold = load_layout(p), load_layout(d / "layout.gold.json")
        r = compare(pred, gold, (d / "report.he.txt").read_text(encoding="utf-8"))
        m = dict(r["matches"])
        P, G = {e["id"]: e for e in pred["elements"]}, {e["id"]: e for e in gold["elements"]}
        rp = {(m.get(x["subject"]), x["predicate"], m.get(x["object"])) for x in pred["relations"]}
        rg = {(x["subject"], x["predicate"], x["object"]) for x in gold["relations"]}
        strict += [len(rp & rg), len(rp), len(rg)]
        cp, cg = {canon(x) for x in rp}, {canon(x) for x in rg}
        can += [len(cp & cg), len(cp), len(cg)]
        for x in cg - cp:
            miss[x[1]] += 1
        for x in cp - cg:
            extra[x[1]] += 1
        for a, b in r["matches"]:
            if G[b]["position"] == "unspecified" or P[a]["position"] == "unspecified":
                continue
            A, B = P[a]["bbox"], G[b]["bbox"]
            c = np.hypot(A["x"] + A["w"] / 2 - B["x"] - B["w"] / 2, A["y"] + A["h"] / 2 - B["y"] - B["h"] / 2)
            ix = max(0, min(A["x"] + A["w"], B["x"] + B["w"]) - max(A["x"], B["x"]))
            iy = max(0, min(A["y"] + A["h"], B["y"] + B["h"]) - max(A["y"], B["y"]))
            inter = ix * iy
            iou = inter / (A["w"] * A["h"] + B["w"] * B["h"] - inter)
            dist[G[b]["class"]].append(float(c))
            ious[G[b]["class"]].append(float(iou))
    if not n:
        return None
    allc, alli = sum(dist.values(), []), sum(ious.values(), [])
    pr = lambda t: {"tp": int(t[0]), "pred": int(t[1]), "gold": int(t[2]), "precision": round(t[0] / t[1], 5), "recall": round(t[0] / t[2], 5)}  # noqa: E731
    return {"n_reports": n, "relations_strict": pr(strict), "relations_canonical": pr(can), "missed_relations": dict(miss.most_common()), "extra_relations": dict(extra.most_common()),
            "centre_offset_m": {"median": round(float(np.median(allc)), 3), "p90": round(float(np.percentile(allc, 90)), 3), "max": round(float(np.max(allc)), 3),
                                **{c: round(float(np.median(v)), 3) for c, v in sorted(dist.items())}},
            "box_iou": {"mean": round(float(np.mean(alli)), 3), **{c: round(float(np.mean(v)), 3) for c, v in sorted(ious.items())}}}


# ------------------------------------------------------------------ סגמנטציה על צילומים אמיתיים

SURFACES = ["vegetation", "shadow", "bright stone or rock", "soil and rubble"]


def surface_map(a: np.ndarray) -> np.ndarray:
    """חלוקה גסה של הפיקסלים לפי צבע, כדי לדעת על מה המודל טועה."""
    f = a.astype(np.float32) / 255
    v = f.mean(axis=2)
    out = np.full(v.shape, 3, np.uint8)
    out[v > 0.58] = 2
    out[(f[..., 1] - np.maximum(f[..., 0], f[..., 2])) > 0.02] = 0
    out[v < 0.18] = 1
    return out


def real_walls() -> dict | None:
    base = PROJECT / "results" / "predictions" / "real_walls"
    if not base.exists():
        return None
    out = {}
    for mdir in sorted(p for p in base.iterdir() if p.is_dir()):
        fp, fn, wall, area = (np.zeros(4) for _ in range(4))
        for p in sorted(mdir.glob("*.png")):
            im = np.array(Image.open(REAL_DIR / "walls" / "test" / "images" / p.name).convert("RGB"))
            gt = np.array(Image.open(REAL_DIR / "walls" / "test" / "masks" / p.name))
            pr = np.array(Image.open(p)) > 127
            s, ok = surface_map(im), gt != IGNORE
            for k in range(4):
                m = (s == k) & ok
                area[k] += m.sum()
                wall[k] += (m & (gt == 2)).sum()
                fp[k] += (m & pr & (gt != 2)).sum()
                fn[k] += (m & ~pr & (gt == 2)).sum()
        nonwall = area - wall
        out[mdir.name] = {"nonwall_area_share": {n: round(float(v), 3) for n, v in zip(SURFACES, nonwall / nonwall.sum())},
                          "false_positive_rate_by_surface": {n: round(float(a / b), 3) if b else None for n, a, b in zip(SURFACES, fp, nonwall)},
                          "false_positive_share": {n: round(float(v), 3) for n, v in zip(SURFACES, fp / max(fp.sum(), 1))},
                          "false_negative_share": {n: round(float(v), 3) for n, v in zip(SURFACES, fn / max(fn.sum(), 1))},
                          "wall_recall_by_surface": {n: round(float(1 - a / b), 3) if b else None for n, a, b in zip(SURFACES, fn, wall)},
                          "wall_share_of_surface": {n: round(float(b / a), 3) if a else None for n, a, b in zip(SURFACES, area, wall)},
                          "wall_pixels_by_surface": {n: round(float(v), 3) for n, v in zip(SURFACES, wall / wall.sum())}}
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    res = {"generation": generation(), "extraction_opus": extraction("layout.pred.json"),
           "extraction_sonnet": extraction("layout.pred_sonnet.json"), "real_walls": real_walls()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "error_analysis.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
