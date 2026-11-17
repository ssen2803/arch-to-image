"""נאמנות מרחבית בשלוש עדשות, ופירוק השגיאה לפי שלבים (מקור: סקירה חיצונית, 17.11.2026).

ההבחנה: נאמנות לדו"ח אינה אותו דבר כמו התאמה לתוכנית. אם בתוכנית יש בור שהדו"ח לא מזכיר,
מערכת שנאמנה לטקסט באופן מושלם תקבל ציון נמוך מול התוכנית. לכן כל תמונה נמדדת שלוש פעמים:

  text_grounded   - מול פריסת gold שנבנתה מהטקסט בלבד: האם המידע שבדו"ח נשמר בתמונה?
  plan_agreement  - מול פריסת הייחוס מהתוכנית: עד כמה התמונה תואמת את התוכנית?
                    התוכנית היא נתון ייחוס גאומטרי, ולא הוכחה למראה המקורי של האתר.
  unsupported     - תוכן בתמונה שאין לו בסיס בטקסט (HOR_area ו-HAR מול פריסת הטקסט).

בנוסף:
  relation_fidelity - איזה חלק מהיחסים המרחביים שבפריסה מתקיים גם בתמונה (בין אלמנטים שנמצאו בה).
  retention         - מדד משני בלבד. מוגדר רק כשהפער בין התוכנית לבין "טקסט בלבד" גדול מסף,
                      ומחושב כיחס ממוצעים על פני מקרים, עם רווח סמך ב-bootstrap. הוא יכול להיות
                      שלילי או גדול מ-1, ולכן מדווח כפי שהוא ולא נחתך.
"""

from __future__ import annotations

import copy

import numpy as np
from scipy import ndimage

from hallucination import COARSE, HallucinationConfig, coarsen, hallucination_report, layout_targets
from mapping import FOREIGN, LABEL, UNKNOWN, project_labels_from_rgb
from layout_core import Box, check_relation, image_transform
from render_layout import render_seg, visibility

DIRECTIONAL_AND_CONTAINMENT = {"north_of", "south_of", "east_of", "west_of", "inside", "on", "adjacent_to"}


def class_map(layout: dict, size: int = 512, frame: dict | None = None) -> np.ndarray:
    """מפת המחלקות של מה שנראה בפריסה. frame: לרנדר במסגרת השטח של פריסה אחרת (להשוואה בין פריסות)."""
    if frame is not None:
        layout = copy.deepcopy(layout)
        layout["area"] = copy.deepcopy(frame["area"])
    return project_labels_from_rgb(np.array(render_seg(layout, size)))


def miou(pred: np.ndarray, ref: np.ndarray, valid: np.ndarray | None = None, granularity: str = "coarse") -> dict:
    """mIoU בין מפת התמונה למפת הייחוס, על המחלקות שמופיעות באחת מהן. פיקסלים לא ידועים לא נספרים."""
    if granularity == "coarse":
        pred, ref = coarsen(pred), coarsen(ref)
    keep = (ref != UNKNOWN) & (pred != UNKNOWN)
    if valid is not None:
        keep &= valid
    classes = sorted(set(np.unique(ref[keep])) | (set(np.unique(pred[keep])) - {FOREIGN}))
    per = {}
    for c in classes:
        p, r = (pred == c) & keep, (ref == c) & keep
        union = (p | r).sum()
        if union:
            per[int(c)] = round(float((p & r).sum() / union), 4)
    return {"miou": round(float(np.mean(list(per.values()))), 4) if per else None, "per_class": per}


def _element_region(pred: np.ndarray, mask: np.ndarray, label: int, min_overlap: float) -> np.ndarray | None:
    """הרכיבים של המחלקה הנכונה בתמונה שנוגעים במסכת האלמנט (רכיבים שלמים, לא חתוכים לתוך המסכה)."""
    comps, _ = ndimage.label(pred == label, structure=np.ones((3, 3), bool))
    ids = np.unique(comps[mask & (comps > 0)])
    if len(ids) == 0:
        return None
    region = np.isin(comps, ids)
    return region if (region & mask).sum() >= min_overlap * mask.sum() else None


def _pixel_box(region: np.ndarray) -> Box:
    ys, xs = np.nonzero(region)
    return Box(float(xs.min()), float(-ys.max()), float(xs.max() + 1), float(-ys.min() + 1))  # y הפוך: צפון למעלה


def relation_fidelity(pred: np.ndarray, layout: dict, size: int = 512, tol_m: float = 0.3,
                      min_overlap: float = 0.3) -> dict:
    """לכל יחס בפריסה בין שני אלמנטים נראים שנמצאו בתמונה: האם הוא מתקיים גם בתמונה.

    pred: מפת תוויות ברזולוציה המלאה (5 מחלקות). כל אלמנט מיוצג ברכיבים מהמחלקה שלו שנוגעים במקומו.
    יחסי cuts ו-under לא נבדקים, כי מלמעלה הם לא נראים כיחס בין שני אזורים.
    """
    from hallucination import element_masks
    vis = visibility(layout, size)
    masks = {m["id"]: m for m in element_masks(layout, size)}
    _, px_per_m = image_transform(layout, size)
    boxes, used, unmatched = {}, set(), []
    comps_by_label = {}
    for eid, m in masks.items():
        if vis.get(eid, {}).get("status") != "visible":
            continue
        region = _element_region(pred, m["mask"], m["label"], min_overlap)
        if region is not None:
            # אלמנטים מאותה מחלקה שנוגעים זה בזה (קירות בפינה) הם רכיב אחד בתמונה. בלי חיתוך לסביבת
            # האלמנט, התיבה של קיר אחד הייתה כוללת גם את הקיר השני. הסביבה: tol_m סביב מקומו המצופה.
            near = ndimage.binary_dilation(m["mask"], iterations=max(1, int(round(tol_m * px_per_m))))
            boxes[eid] = _pixel_box(region & near)
            if m["label"] not in comps_by_label:
                comps_by_label[m["label"]] = ndimage.label(pred == m["label"], structure=np.ones((3, 3), bool))[0]
            used |= {(m["label"], int(i)) for i in np.unique(comps_by_label[m["label"]][region]) if i}
        else:
            unmatched.append(eid)
    # שלב שני: אלמנט שלא נמצא במקומו מותאם לרכיב הקרוב ביותר מאותה מחלקה שעוד לא שויך, אם גודלו סביר.
    # בלי השלב הזה אלמנט שזז רחוק פשוט "לא נמצא", והיחסים שלו לא נבדקים כלל.
    for eid in unmatched:
        m = masks[eid]
        comps = comps_by_label.setdefault(m["label"],
                                          ndimage.label(pred == m["label"], structure=np.ones((3, 3), bool))[0])
        ey, ex = (np.mean(a) for a in np.nonzero(m["mask"]))
        best, best_d = None, np.inf
        for i in np.unique(comps):
            if i == 0 or (m["label"], int(i)) in used:
                continue
            comp = comps == i
            if comp.sum() < min_overlap * m["mask"].sum():
                continue
            cy, cx = (np.mean(a) for a in np.nonzero(comp))
            d = float(np.hypot(cy - ey, cx - ex))
            if d < best_d:
                best, best_d = int(i), d
        if best is not None:
            used.add((m["label"], best))
            boxes[eid] = _pixel_box(comps == best)
    checked = []
    for rel in layout["relations"]:
        s, p, o = rel["subject"], rel["predicate"], rel["object"]
        if p not in DIRECTIONAL_AND_CONTAINMENT or s not in boxes or o not in boxes:
            continue
        ok, _ = check_relation(p, boxes[s], boxes[o], tol_m * px_per_m)
        checked.append({"relation": f"{s} {p} {o}", "holds": bool(ok)})
    n = len(checked)
    return {"rate": round(sum(c["holds"] for c in checked) / n, 4) if n else None, "n_checked": n,
            "n_relations": sum(r["predicate"] in DIRECTIONAL_AND_CONTAINMENT for r in layout["relations"]),
            "checked": checked}


def three_way(pred: np.ndarray, text_layout: dict, plan_layout: dict | None = None, size: int = 512) -> dict:
    """שלוש העדשות לתמונה אחת. pred: מפת התוויות של התמונה (5 מחלקות, מהמעריך)."""
    t = layout_targets(text_layout, size)
    rep = hallucination_report(pred, class_map(text_layout, size), t["elements"],
                               HallucinationConfig(granularity="coarse"), t["valid"], t["uncertain"], t["occluded"])
    out = {
        "text_grounded": {**miou(pred, class_map(text_layout, size), t["valid"]),
                          "object_recall": rep["object_recall"],
                          "relation_fidelity": relation_fidelity(pred, text_layout, size)["rate"]},
        "unsupported": {"HOR": rep["HOR"], "HOR_area": rep["HOR_area"], "HAR": rep["HAR"],
                        "foreign_components": rep["foreign_components"]},
    }
    if plan_layout is not None:
        out["plan_agreement"] = miou(pred, class_map(plan_layout, size, frame=text_layout), t["valid"])
    return out


def layout_miou(layout: dict, reference: dict, size: int = 512, granularity: str = "coarse") -> float | None:
    """התאמה בין שתי פריסות לפני יצירת תמונה (שלב החילוץ), במסגרת השטח של פריסת הייחוס."""
    return miou(class_map(layout, size, frame=reference), class_map(reference, size), granularity=granularity)["miou"]


def retention(s_system: list[float], s_text: list[float], s_plan: list[float], min_gap: float = 0.05,
              n_boot: int = 2000, seed: int = 0) -> dict:
    """Retention = (S_system − S_text) / (S_plan − S_text), כיחס ממוצעים על פני מקרים מזווגים.

    לא מוגדר (value=None) כשהפער הממוצע בין התוכנית לטקסט קטן מ-min_gap: המכנה קרוב לאפס והמדד לא יציב.
    """
    a, b, c = (np.asarray(v, float) for v in (s_system, s_text, s_plan))
    gap = float(np.mean(c - b))
    if len(a) == 0 or gap < min_gap:
        return {"value": None, "gap": round(gap, 4) if len(a) else None,
                "reason": f"הפער בין התוכנית לטקסט ({gap:.3f}) קטן מ-{min_gap}; המדד לא יציב" if len(a) else "אין מקרים"}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(a), len(a))
        g = np.mean(c[i] - b[i])
        if g >= min_gap / 2:
            vals.append(float(np.mean(a[i] - b[i]) / g))
    lo, hi = (np.quantile(vals, [0.025, 0.975]) if vals else (np.nan, np.nan))
    return {"value": round(float(np.mean(a - b) / gap), 4), "gap": round(gap, 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)], "n_cases": len(a)}


__all__ = ["COARSE", "LABEL", "class_map", "miou", "relation_fidelity", "three_way", "layout_miou", "retention"]
