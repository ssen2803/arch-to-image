"""EDA של הנתונים הסינתטיים (600 תמונות) ושל אריחי הרחפן האמיתיים, לפי שמונת ההיעדים של מדריך הקורס:
תקינות, כיסוי, ריאליזם, גיוון, איכות תיוג, דליפה, איזון, קושי.

פלט: results/eda/eda.json (כל המספרים) ו-results/eda/eda_raw.json (ההתפלגויות, לתרשימים במחברת ובמצגת).

ההשוואה לצילומים אמיתיים נעשית על 14 אריחי dev בלבד, כדי שהחלטות שנגזרות ממנה לא יישענו על אריחי ה-test.

שימוש:
    python src/eda.py
"""

from __future__ import annotations

import collections
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES, IGNORE, PROJECT, REAL_DIR, ROOT, SYNTH_DIR  # noqa: E402
from compositions import ALL_TAGS, composition_tags  # noqa: E402
from datasets import drop_blank_images, read_manifest, split_by_layout3  # noqa: E402

OUT = PROJECT / "results" / "eda"


def gray64(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L").resize((64, 64), Image.BILINEAR), dtype=np.float32)


def dhash(path, n=8) -> int:
    a = np.asarray(Image.open(path).convert("L").resize((n + 1, n), Image.BILINEAR), dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def low_level(path) -> dict:
    """סטטיסטיקות פשוטות להשוואה בין סינתטי לאמיתי: בהירות, ניגודיות, רוויה, וחדות (שונות הלפלסיאן)."""
    im = Image.open(path).convert("RGB").resize((512, 512), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32) / 255
    g = a.mean(axis=2)
    valid = a.max(axis=2) > 0.012
    mx, mn = a.max(axis=2), a.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    green = (a[..., 1] - np.maximum(a[..., 0], a[..., 2])) > 0.02
    return {"brightness": float(g[valid].mean()), "contrast": float(g[valid].std()),
            "saturation": float(sat[valid].mean()), "sharpness": float(ndimage.laplace(g)[valid].var()),
            "green_frac": float(green[valid].mean())}


def wall_thickness_px(mask: np.ndarray) -> float | None:
    """עובי קיר אופייני: כפול הערך החציוני של מרחק השלד מהקצה. ביחס לצלע התמונה."""
    if mask.sum() < 200:
        return None
    d = ndimage.distance_transform_edt(mask)
    ridge = (d == ndimage.maximum_filter(d, size=5)) & (d > 1)
    return float(2 * np.median(d[ridge]) / mask.shape[0]) if ridge.any() else None


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(SYNTH_DIR)
    E: dict = {"n_images": len(rows), "n_layouts": len({r["layout_id"] for r in rows})}

    # 1. Sanity
    sizes = collections.Counter(Image.open(r["image_path"]).size for r in rows)
    missing = {k: sum(r[k] in (None, "") for r in rows) for k in rows[0] if k not in ("image_path", "label_path")}
    md5 = collections.Counter(hashlib.md5(Path(r["image_path"]).read_bytes()).hexdigest() for r in rows)
    good, blank = drop_blank_images(rows)
    labs = {}
    bad_vals = 0
    for r in rows:
        if r["label_path"] not in labs:
            labs[r["label_path"]] = np.array(Image.open(r["label_path"]))
            bad_vals += int((labs[r["label_path"]] >= len(CLASSES)).any())
    E["sanity"] = {"image_sizes": {str(k): v for k, v in sizes.items()}, "missing_values": {k: v for k, v in missing.items() if v},
                   "exact_duplicate_images": sum(v - 1 for v in md5.values() if v > 1), "blank_images": [r["image"] for r in blank],
                   "labels_with_invalid_values": bad_vals}

    # 2 + 7. Coverage, balance
    px = np.zeros(len(CLASSES))
    pres = np.zeros(len(CLASSES), int)
    tags = collections.Counter()
    per_layout_tags, comp_n = {}, {c: [] for c in CLASSES[1:]}
    thick = []
    for p, lab in labs.items():
        px += np.bincount(lab.ravel(), minlength=len(CLASSES))[:len(CLASSES)]
        pres += np.array([(lab == i).any() for i in range(len(CLASSES))])
        t = composition_tags(lab)
        per_layout_tags[p] = t
        tags.update(t)
        for i, c in enumerate(CLASSES[1:], start=1):
            comp_n[c].append(int(ndimage.label(lab == i)[1]))
        th = wall_thickness_px(lab == 2)
        if th:
            thick.append(th)
    layouts = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((SYNTH_DIR / "layouts").glob("*.json"))]
    terms, periods, contexts, shapes = (collections.Counter() for _ in range(4))
    for l in layouts:
        periods[l["source"]["period"]] += 1
        contexts[l["source"]["context"]] += 1
        for e in l["elements"]:
            terms[(e["class"], e["subtype"])] += 1
    E["coverage"] = {
        "pixel_share": {c: round(float(v), 4) for c, v in zip(CLASSES, px / px.sum())},
        "layouts_with_class": {c: int(v) for c, v in zip(CLASSES, pres)},
        "elements_per_class": {c: int(sum(v for (k, _), v in terms.items() if k == c)) for c in CLASSES[1:]},
        "element_types": {f"{k[0]}/{k[1]}": v for k, v in terms.most_common()},
        "elements_per_layout": {"mean": round(float(np.mean([len(l["elements"]) for l in layouts])), 2),
                                "min": min(len(l["elements"]) for l in layouts), "max": max(len(l["elements"]) for l in layouts)},
        "components_per_mask": {c: {"mean": round(float(np.mean(v)), 2), "max": int(max(v))} for c, v in comp_n.items()},
        "periods": dict(periods.most_common()), "contexts": dict(contexts.most_common()),
        "composition_tags": {t: tags[t] for t in ALL_TAGS}, "empty_layouts": int(sum(len(l["elements"]) == 0 for l in layouts)),
    }

    # 6. Leakage
    tr, va, te = split_by_layout3(good)
    ids = [{r["layout_id"] for r in s} for s in (tr, va, te)]
    h = {r["image"]: dhash(r["image_path"]) for r in good}
    split_of = {r["image"]: n for n, s in (("train", tr), ("val", va), ("test", te)) for r in s}
    names = sorted(h)
    cross = near = 0
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if bin(h[a] ^ h[b]).count("1") <= 6:
                near += 1
                cross += split_of[a] != split_of[b]
    lab_hash = collections.defaultdict(set)
    for r in good:
        lab_hash[hashlib.md5(labs[r["label_path"]].tobytes()).hexdigest()].add(r["layout_id"])
    E["leakage"] = {"split_images": {"train": len(tr), "val": len(va), "test": len(te)},
                    "split_layouts": {"train": len(ids[0]), "val": len(ids[1]), "test": len(ids[2])},
                    "layouts_shared_between_splits": len((ids[0] & ids[1]) | (ids[0] & ids[2]) | (ids[1] & ids[2])),
                    "identical_masks_across_layouts": sum(len(v) > 1 for v in lab_hash.values()),
                    "near_duplicate_pairs_dhash<=6": near, "near_duplicate_pairs_across_splits": cross}

    # 4. Variability: מה קובע את המראה, הפריסה או ה-seed של הדיפוזיה?
    g = {r["image"]: gray64(r["image_path"]) for r in good}
    keys = sorted(g)
    X = np.stack([((g[k] - g[k].mean()) / (g[k].std() + 1e-6)).ravel() for k in keys])
    C = X @ X.T / X.shape[1]
    meta = {r["image"]: (r["layout_id"], int(r["seed"]) % 1000) for r in good}
    lid, sd = np.array([meta[k][0] for k in keys]), np.array([meta[k][1] for k in keys])
    iu = np.triu_indices(len(keys), 1)
    sl, ss = lid[iu[0]] == lid[iu[1]], sd[iu[0]] == sd[iu[1]]
    corr = {"same_layout_diff_seed": C[iu][sl & ~ss], "diff_layout_same_seed": C[iu][~sl & ss],
            "diff_layout_diff_seed": C[iu][~sl & ~ss]}
    E["variability"] = {"pixel_correlation_64px": {k: {"median": round(float(np.median(v)), 3), "n_pairs": int(v.size)}
                                                   for k, v in corr.items()},
                        "n_diffusion_seeds": int(len(set(sd))), "unique_prompts": len({r["prompt"] for r in rows})}

    # 5. Label quality: הסכמת התיוג עם מעריך חיצוני, ועם חיזוי מחוץ לקיפול
    agree = np.array([r["agree_miou"] for r in good if r["agree_miou"] is not None])
    oof = {}
    cf = ROOT / "runs" / "crossfit" / "scores.csv"
    if cf.exists():
        with open(cf, encoding="utf-8") as f:
            oof = {r["image"]: float(r["oof_miou"]) for r in csv.DictReader(f)}
    E["label_quality"] = {"segformer_agreement": {"median": round(float(np.median(agree)), 3),
                                                  **{f"share>={t}": round(float((agree >= t).mean()), 4) for t in (0.3, 0.5, 0.8)}},
                          "HAR_median": round(float(np.median([r["HAR"] for r in good if r["HAR"] is not None])), 3),
                          "crossfit_oof_miou": {"median": round(float(np.median(list(oof.values()))), 3), "n": len(oof)} if oof else None}

    # 3. Realism: סינתטי מול אמיתי
    real = sorted((REAL_DIR / "walls").glob("*/images/*.png"))
    real = [p for p in real if p.parts[-3] == "dev"]
    sub = good[::max(1, len(good) // 200)][:200]
    S = [low_level(r["image_path"]) for r in sub]
    R = [low_level(p) for p in real]
    rthick = []
    rfrac = []
    for p in real:
        m = np.array(Image.open(p.parent.parent / "masks" / p.name))
        rfrac.append(float((m == 2).sum() / max(1, (m != IGNORE).sum())))
        t = wall_thickness_px(m == 2)
        if t:
            rthick.append(t)
    sfrac = [float((l == 2).mean()) for l in labs.values()]
    from scipy.stats import ks_2samp
    E["realism"] = {"n_synth": len(S), "n_real": len(R), "stats": {}}
    for k in S[0]:
        a, b = np.array([x[k] for x in S]), np.array([x[k] for x in R])
        ks = ks_2samp(a, b)
        E["realism"]["stats"][k] = {"synth_median": round(float(np.median(a)), 4), "real_median": round(float(np.median(b)), 4),
                                    "ks": round(float(ks.statistic), 3)}
    E["realism"]["wall_thickness_share_of_side"] = {"synth_median": round(float(np.median(thick)), 4), "real_median": round(float(np.median(rthick)), 4)}
    E["realism"]["wall_pixel_share"] = {"synth_median": round(float(np.median(sfrac)), 4), "real_median": round(float(np.median(rfrac)), 4)}
    raw = {"synth": S, "real": R, "thick_synth": thick, "thick_real": rthick, "wallfrac_synth": sfrac, "wallfrac_real": rfrac,
           "agree": agree.tolist(), "oof": list(oof.values()), "corr": {k: np.random.default_rng(0).choice(v, min(1500, v.size), replace=False).round(4).tolist()
                                                        for k, v in corr.items()},
           "elements_per_layout": [len(l["elements"]) for l in layouts]}
    (OUT / "eda.json").write_text(json.dumps(E, indent=1, ensure_ascii=False), encoding="utf-8")
    (OUT / "eda_raw.json").write_text(json.dumps(raw), encoding="utf-8")
    print(json.dumps(E, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
