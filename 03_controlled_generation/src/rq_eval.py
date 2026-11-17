"""הערכת שאלת המחקר המרכזית: טקסט בלבד מול פריסה מ-LLM מול פריסה מהתוכנית.

שאלת המחקר: באיזו מידה פריסה מרחבית מובנית, שחולצה בידי מודל שפה מדו"ח חפירה, משפרת את הנאמנות
המרחבית של תמונות גנרטיביות לעומת יצירה מטקסט בלבד, וכיצד היא משתווה להנחיה בפריסה מהתוכנית המקורית?

הניסוי מוגבל למקרים שיש להם גם תיאור טקסטואלי וגם תוכנית תואמת. מבנה תיקיית מקרה:

    <cases>/<case_id>/report.he.txt
    <cases>/<case_id>/layout.text_gold.json   חובה: פריסה שמתייג בנה מהטקסט בלבד
    <cases>/<case_id>/layout.plan.json        פריסת ייחוס מהתוכנית (בלעדיה אין plan_agreement)
    <cases>/<case_id>/layout.llm.json         הפריסה שחילץ מודל השפה
    <cases>/<case_id>/images/{text_only,llm_layout,plan_layout}_s<seed>.png

פירוק השגיאה לשלבים, כדי לדעת איפה המידע אבד:
  1. חילוץ:  layout.llm מול layout.text_gold (compare_layouts), ושתיהן מול layout.plan (layout_miou).
             layout.text_gold מול layout.plan מראה כמה מהתוכנית בכלל נמצא בטקסט.
  2. ייצור:  כל תמונה מול הפריסה שהוזנה לה (HOR, Recall): האם ControlNet שמר על מה שקיבל?
  3. סופי:   כל תמונה בשלוש העדשות של fidelity.three_way.

מדדים ראשיים: mIoU מול התוכנית ומול פריסת הטקסט, ונאמנות היחסים. Retention מדווח כמדד משני בלבד.

שימוש:
    python src/rq_eval.py --cases data/rq_cases --out outputs/rq
    python src/rq_eval.py --cases data/rq_cases --evaluator palette     # לתמונות בפלטת הפרויקט (בדיקות)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fidelity import layout_miou, retention, three_way  # noqa: E402
from hallucination import HallucinationConfig, hallucination_report, layout_targets  # noqa: E402
from fidelity import class_map  # noqa: E402
from compare_layouts import compare  # noqa: E402
from layout_core import load_layout  # noqa: E402

CONDITIONS = {"text_only": None, "llm_layout": "layout.llm.json", "plan_layout": "layout.plan.json"}


def load_case(case: Path) -> dict:
    get = lambda name: load_layout(case / name) if (case / name).exists() else None  # noqa: E731
    text_path = case / "report.he.txt"
    return {"id": case.name, "text": text_path.read_text(encoding="utf-8") if text_path.exists() else "",
            "text_gold": get("layout.text_gold.json"), "plan": get("layout.plan.json"), "llm": get("layout.llm.json")}


def extraction_stage(c: dict) -> dict:
    out = {}
    if c["llm"] is not None:
        r = compare(c["llm"], c["text_gold"], c["text"])
        out["llm_vs_text_gold"] = {k: r[k] for k in ("element_precision", "element_recall", "box_iou",
                                                     "positional_invention_rate", "over_specification_rate",
                                                     "relation_recall")}
    if c["plan"] is not None:
        out["text_gold_vs_plan_miou"] = layout_miou(c["text_gold"], c["plan"])   # כמה מהתוכנית נמצא בטקסט
        if c["llm"] is not None:
            out["llm_vs_plan_miou"] = layout_miou(c["llm"], c["plan"])
    return out


def evaluate_case(case: Path, evaluator, size: int = 512) -> dict:
    c = load_case(case)
    if c["text_gold"] is None:
        raise FileNotFoundError(f"{case.name}: חסר layout.text_gold.json")
    result = {"case": c["id"], "extraction": extraction_stage(c), "images": []}
    for img in sorted((case / "images").glob("*_s[0-9]*.png")):
        cond = img.stem.rsplit("_s", 1)[0]
        if cond not in CONDITIONS:
            continue
        pred = evaluator(np.array(Image.open(img).convert("RGB").resize((size, size))))
        entry = {"condition": cond, "seed": int(img.stem.rsplit("_s", 1)[1]),
                 **three_way(pred, c["text_gold"], c["plan"], size)}
        given = {"llm_layout": c["llm"], "plan_layout": c["plan"]}.get(cond)
        if given is not None:                                    # שלב הייצור: מול הפריסה שהוזנה
            t = layout_targets(given, size)
            rep = hallucination_report(pred, class_map(given, size), t["elements"],
                                       HallucinationConfig(granularity="coarse"), t["valid"], t["uncertain"],
                                       t["occluded"])
            entry["generation_vs_given_layout"] = {"HOR": rep["HOR"], "object_recall": rep["object_recall"]}
        result["images"].append(entry)
    return result


def _case_mean(results: list[dict], cond: str, lens: str, key: str) -> dict[str, float]:
    """ממוצע על seeds לכל מקרה. המקרה הוא יחידת הניתוח."""
    out = {}
    for r in results:
        vals = [im[lens][key] for im in r["images"] if im["condition"] == cond and lens in im
                and im[lens].get(key) is not None]
        if vals:
            out[r["case"]] = float(np.mean(vals))
    return out


def aggregate(results: list[dict], min_gap: float = 0.05) -> dict:
    summary = {}
    metrics = [("plan_agreement", "miou"), ("text_grounded", "miou"), ("text_grounded", "object_recall"),
               ("text_grounded", "relation_fidelity"), ("unsupported", "HOR_area"), ("unsupported", "HAR")]
    for cond in CONDITIONS:
        summary[cond] = {}
        for lens, key in metrics:
            per_case = _case_mean(results, cond, lens, key)
            if per_case:
                summary[cond][f"{lens}.{key}"] = {"mean": round(statistics.mean(per_case.values()), 4),
                                                  "n_cases": len(per_case)}
    plan = {c: _case_mean(results, c, "plan_agreement", "miou") for c in CONDITIONS}
    common = sorted(set(plan["text_only"]) & set(plan["llm_layout"]) & set(plan["plan_layout"]))
    summary["retention_plan_miou"] = retention([plan["llm_layout"][k] for k in common],
                                               [plan["text_only"][k] for k in common],
                                               [plan["plan_layout"][k] for k in common], min_gap=min_gap)
    return summary


def to_markdown(summary: dict) -> str:
    rows = ["| מדד | text_only | llm_layout | plan_layout |", "|---|---|---|---|"]
    keys = sorted({k for c in CONDITIONS for k in summary.get(c, {})})
    for k in keys:
        cells = [f"{summary[c][k]['mean']:.3f} (n={summary[c][k]['n_cases']})" if k in summary.get(c, {}) else "—"
                 for c in CONDITIONS]
        rows.append(f"| {k} | " + " | ".join(cells) + " |")
    r = summary["retention_plan_miou"]
    ret = (f"Retention (מדד משני) = {r['value']} (95% CI {r['ci95'][0]}–{r['ci95'][1]}, {r['n_cases']} מקרים)"
           if r["value"] is not None else f"Retention לא מוגדר: {r['reason']}")
    return "# שאלת המחקר: שלושה תנאים\n\n" + "\n".join(rows) + f"\n\n{ret}\n"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="הערכת שאלת המחקר: טקסט בלבד, פריסה מ-LLM, פריסה מהתוכנית")
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/rq"))
    ap.add_argument("--evaluator", choices=["segformer", "palette"], default="segformer")
    ap.add_argument("--min-gap", type=float, default=0.05, help="פער מינימלי בין התוכנית לטקסט כדי ש-Retention יוגדר")
    args = ap.parse_args()

    if args.evaluator == "palette":
        from hallucination import PaletteEvaluator
        evaluator = PaletteEvaluator()
    else:
        from hallucination import SegformerEvaluator
        from mapping import load_mapping
        evaluator = SegformerEvaluator(load_mapping())
    cases = sorted(p for p in args.cases.iterdir() if p.is_dir())
    results = [evaluate_case(c, evaluator) for c in cases]
    summary = aggregate(results, args.min_gap)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps({"cases": results, "summary": summary}, ensure_ascii=False,
                                                      indent=1), encoding="utf-8")
    (args.out / "results.md").write_text(to_markdown(summary), encoding="utf-8")
    print(to_markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
