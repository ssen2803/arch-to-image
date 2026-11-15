"""השוואת Layout שחולץ (pred) מול Layout gold: כמה המחלץ המציא, ומה הוא פספס.

התאמת אלמנטים: לפי חפיפת התווים של הציטוטים (source_span) בתוך טקסט המקור.
שני ה-Layouts מצטטים מילה במילה מאותו טקסט, ולכן ההתאמה לא תלויה בגיאומטריה:
אלמנט שהוצב במקום הלא נכון עדיין יותאם, והטעות שלו תימדד בנפרד ב-IoU.

מדדים עיקריים:
  element_invention_rate   - אלמנטים שחולצו ואין להם מקבילה ב-gold (המצאת קיום)
  positional_invention_rate - מבין האלמנטים שב-gold מיקומם unspecified, כמה קיבלו מיקום מוגדר
                              (stated/relative) - כלומר המחלץ "המציא" היכן הם נמצאים
  box_iou                   - IoU ממוצע של תיבות (במטרים) לאלמנטים שמיקומם ידוע בשניהם
  occlusion_recall          - מבין האלמנטים המוסתרים ב-gold, כמה המחלץ גם הסתיר (גזם)
  false_occlusion_rate      - מבין האלמנטים הנראים ב-gold, כמה המחלץ הסתיר בטעות
  relation_precision/recall - על שלשות (נושא, יחס, מושא) אחרי מיפוי המזהים

מדדי מונחים (סכמה 1.2, רק לזוגות שיש בהם term בשני הצדדים):
  term_exact                - המונח זהה
  term_hier                 - דמיון היררכי: חפיפת שרשראות האבות (Jaccard), כך שאב נכון מקבל קרדיט חלקי
  over_specification_rate   - המונח החזוי צאצא ממש של מונח ה-gold: המחלץ אמר יותר ממה שהטקסט תומך
  under_specification_rate  - המונח החזוי אב ממש של מונח ה-gold: המחלץ אמר פחות (זהיר מדי, לא שגוי)

שימוש:
    python src/compare_layouts.py --pred examples/ex01_iron_age_room/layout.pred.json \
                                  --gold examples/ex01_iron_age_room/layout.gold.json
    python src/compare_layouts.py --all     # כל examples/*/layout.pred.json מול ה-gold שלהן
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layout_core import ROOT, Box, find_report_text, load_layout, term_ancestors  # noqa: E402


def _interval(text: str, span: str) -> tuple[int, int] | None:
    norm = " ".join(text.split())
    s = " ".join(span.split())
    i = norm.find(s)
    return (i, i + len(s)) if i >= 0 else None


def _span_overlap(a: tuple[int, int] | None, b: tuple[int, int] | None) -> float:
    if a is None or b is None:
        return 0.0
    inter = min(a[1], b[1]) - max(a[0], b[0])
    return max(0.0, inter) / min(a[1] - a[0], b[1] - b[0])


def _box_iou(a: dict, b: dict) -> float:
    A, B = Box.from_bbox(a), Box.from_bbox(b)
    inter = A.intersection_area(B)
    return inter / (A.area + B.area - inter) if inter else 0.0


def match_elements(pred: dict, gold: dict, text: str, min_overlap: float = 0.3) -> list[tuple[str, str]]:
    """התאמה אחד-לאחד (Hungarian) לפי חפיפת ציטוטים; בהיעדר ציטוט - לפי IoU של תיבות באותה מחלקה."""
    from scipy.optimize import linear_sum_assignment
    P, G = pred["elements"], gold["elements"]
    if not P or not G:
        return []
    score = np.zeros((len(P), len(G)))
    for i, p in enumerate(P):
        ip = _interval(text, p["source_span"]) if text else None
        for j, g in enumerate(G):
            ov = _span_overlap(ip, _interval(text, g["source_span"])) if text else 0.0
            if ov == 0.0 and p["class"] == g["class"]:
                ov = 0.5 * _box_iou(p["bbox"], g["bbox"])  # גיבוי גיאומטרי, במשקל נמוך
            score[i, j] = ov
    rows, cols = linear_sum_assignment(-score)
    return [(P[r]["id"], G[c]["id"]) for r, c in zip(rows, cols) if score[r, c] >= min_overlap]


def compare(pred: dict, gold: dict, text: str = "") -> dict:
    from render_layout import visibility
    pairs = match_elements(pred, gold, text)
    pmap = {p: g for p, g in pairs}
    P = {e["id"]: e for e in pred["elements"]}
    G = {e["id"]: e for e in gold["elements"]}

    gold_unspec = [g for p, g in pairs if G[g]["position"] == "unspecified"]
    invented_pos = [g for p, g in pairs if G[g]["position"] == "unspecified" and P[p]["position"] != "unspecified"]
    ious = [_box_iou(P[p]["bbox"], G[g]["bbox"]) for p, g in pairs
            if G[g]["position"] != "unspecified" and P[p]["position"] != "unspecified"]

    vp, vg = visibility(pred), visibility(gold)
    gold_occ = [g for g, v in vg.items() if v["status"] == "occluded"]
    gold_vis = [g for g, v in vg.items() if v["status"] == "visible"]
    inv = {g: p for p, g in pairs}
    occ_hit = [g for g in gold_occ if g in inv and vp.get(inv[g], {}).get("status") == "occluded"]
    false_occ = [g for g in gold_vis if g in inv and vp.get(inv[g], {}).get("status") == "occluded"]

    rel_p = {(pmap.get(r["subject"]), r["predicate"], pmap.get(r["object"])) for r in pred["relations"]}
    rel_g = {(r["subject"], r["predicate"], r["object"]) for r in gold["relations"]}
    rel_tp = len(rel_p & rel_g)

    def ratio(a, b):
        return round(a / b, 4) if b else None

    tpairs = [(P[p]["term"]["value"], G[g]["term"]["value"]) for p, g in pairs if "term" in P[p] and "term" in G[g]]
    over = under = 0
    hier = []
    for tp, tg in tpairs:
        ap, ag = term_ancestors(tp), term_ancestors(tg)
        hier.append(len(set(ap) & set(ag)) / len(set(ap) | set(ag)))
        over += tp != tg and tg in ap
        under += tp != tg and tp in ag

    return {
        "n_pred": len(P), "n_gold": len(G), "n_matched": len(pairs),
        "element_precision": ratio(len(pairs), len(P)),
        "element_recall": ratio(len(pairs), len(G)),
        "element_invention_rate": ratio(len(P) - len(pairs), len(P)),
        "invented_elements": sorted(set(P) - set(pmap)),
        "missed_elements": sorted(set(G) - set(inv)),
        "class_accuracy": ratio(sum(P[p]["class"] == G[g]["class"] for p, g in pairs), len(pairs)),
        "positional_invention_rate": ratio(len(invented_pos), len(gold_unspec)),
        "positionally_invented": invented_pos,
        "box_iou": round(statistics.mean(ious), 4) if ious else None,
        "occlusion_recall": ratio(len(occ_hit), len(gold_occ)),
        "false_occlusion_rate": ratio(len(false_occ), len(gold_vis)),
        "relation_precision": ratio(rel_tp, len(rel_p)),
        "relation_recall": ratio(rel_tp, len(rel_g)),
        "n_term_pairs": len(tpairs),
        "term_exact": ratio(sum(tp == tg for tp, tg in tpairs), len(tpairs)),
        "term_hier": round(statistics.mean(hier), 4) if hier else None,
        "over_specification_rate": ratio(over, len(tpairs)),
        "under_specification_rate": ratio(under, len(tpairs)),
        "matches": pairs,
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="השוואת Layout שחולץ מול gold")
    ap.add_argument("--pred", type=Path)
    ap.add_argument("--gold", type=Path)
    ap.add_argument("--all", action="store_true", help="כל examples/*/layout.pred.json")
    args = ap.parse_args()
    pairs = ([(p, p.parent / "layout.gold.json") for p in sorted((ROOT / "examples").glob("*/layout.pred.json"))]
             if args.all else [(args.pred, args.gold)])
    if not pairs or pairs[0][0] is None:
        ap.error("צריך --pred ו---gold, או --all")
    for pred_path, gold_path in pairs:
        text_path = find_report_text(gold_path)
        text = text_path.read_text(encoding="utf-8") if text_path else ""
        res = compare(load_layout(pred_path), load_layout(gold_path), text)
        res.pop("matches")
        print(f"{pred_path.parent.name}: " + json.dumps(res, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
