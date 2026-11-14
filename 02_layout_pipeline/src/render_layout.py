"""רינדור Layout לשלושה תוצרים:

1. <name>.schematic.png - איור סכמטי מקודד-ראיות (ההכרעה משלב 1):
   מילוי לפי מחלקה, ורמת הוודאות מקודדת בסגנון הקו והשקיפות:
   attested = קו רציף ומילוי מלא, inferred = קו מקווקו ומילוי חלקי,
   hypothetical = קו מנוקד ומילוי "רפאים".
2. <name>.seg.png - מסכת סגמנטציה בצבעים שטוחים מהפלטה הקבועה
   (ללא קווים/תוויות). משמשת כקלט ל-ControlNet וכ-ground truth ל-mIoU.
3. <name>.boxes.json - תיבות מנורמלות + phrases בפורמט GLIGEN, וכותרת גלובלית.

שימוש:
    python src/render_layout.py examples/ex01_iron_age_room/layout.gold.json
    python src/render_layout.py examples/*/layout.gold.json --size 512
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Ellipse, Patch, Rectangle  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layout_core import (  # noqa: E402
    CLASSES, PALETTE, Z_ORDER, Box, errors_only, global_caption, image_boxes, image_transform,
    load_layout, stratigraphic_order, uncertainty_box, validate,
)

MIN_VISIBLE = 0.05  # אלמנט שפחות מ-5% ממנו נראה במבט-על נחשב מוסתר (נגזם מה-Layout הנראה)


def drawn_elements(layout: dict) -> list[dict]:
    """האלמנטים שמצוירים במקום מוגדר, מלמטה למעלה בסדר סטרטיגרפי. אלמנט בלי מיקום
    (position=unspecified) לא מצויר בנקודה מומצאת - הוא מוצג כאזור אי-ודאות."""
    order = stratigraphic_order(layout) or sorted(layout["elements"], key=lambda e: Z_ORDER[e["class"]])
    return [e for e in order if e["position"] != "unspecified"]


def _px_rect(to_px, el: dict) -> list[int]:
    b = Box.from_bbox(el["bbox"])
    x0, y0 = to_px(b.x0, b.y1)
    x1, y1 = to_px(b.x1, b.y0)
    return [round(x0), round(y0), round(x1) - 1, round(y1) - 1]


def render_id_map(layout: dict, size: int = 512) -> tuple[np.ndarray, list[dict]]:
    """מפת מזהים: 0 = רקע, k = האלמנט ה-k (מבוסס 1) ב-drawn_elements. אלמנט עליון מסתיר תחתון."""
    els = drawn_elements(layout)
    img = Image.new("I", (size, size), 0)
    draw = ImageDraw.Draw(img)
    to_px, _ = image_transform(layout, size)
    for k, el in enumerate(els, start=1):
        (draw.ellipse if el["shape"] == "ellipse" else draw.rectangle)(_px_rect(to_px, el), fill=k)
    return np.array(img, dtype=np.int32), els


def visibility(layout: dict, size: int = 512) -> dict[str, dict]:
    """לכל אלמנט: איזה חלק ממנו נראה במבט-על, ומעמדו: visible / occluded / unspecified."""
    ids, els = render_id_map(layout, size)
    out = {e["id"]: {"visible_fraction": None, "status": "unspecified"}
           for e in layout["elements"] if e["position"] == "unspecified"}
    to_px, _ = image_transform(layout, size)
    for k, el in enumerate(els, start=1):
        alone = Image.new("1", (size, size), 0)
        (ImageDraw.Draw(alone).ellipse if el["shape"] == "ellipse" else ImageDraw.Draw(alone).rectangle)(
            _px_rect(to_px, el), fill=1)
        total = int(np.array(alone).sum())
        frac = float((ids == k).sum() / total) if total else 0.0
        out[el["id"]] = {"visible_fraction": round(frac, 3),
                         "status": "visible" if frac >= MIN_VISIBLE else "occluded"}
    return out

EVIDENCE_STYLE = {
    "attested": {"fill_alpha": 1.0, "linestyle": "-", "linewidth": 1.4},
    "inferred": {"fill_alpha": 0.5, "linestyle": (0, (5, 3)), "linewidth": 1.4},
    "hypothetical": {"fill_alpha": 0.18, "linestyle": (0, (1, 2.5)), "linewidth": 1.6},
}


def _rgb(c: tuple[int, int, int]) -> tuple[float, float, float]:
    return tuple(v / 255 for v in c)


def render_seg(layout: dict, size: int = 512) -> Image.Image:
    """מסכת המחלקות של מה שנראה במבט-על: סדר סטרטיגרפי, בלי אלמנטים מוסתרים או בלי מיקום."""
    ids, els = render_id_map(layout, size)
    lut = np.array([PALETTE["background"]] + [PALETTE[e["class"]] for e in els], dtype=np.uint8)
    return Image.fromarray(lut[ids])


def render_schematic(layout: dict, out_path: Path) -> None:
    W, H = layout["area"]["width_m"], layout["area"]["height_m"]
    fig, ax = plt.subplots(figsize=(7.5, 7.5 * max(H / W, 0.55) + 1.2))
    fig.patch.set_facecolor(_rgb(PALETTE["background"]))
    ax.set_facecolor(_rgb(PALETTE["background"]))

    # גבול החפירה
    ax.add_patch(Rectangle((0, 0), W, H, fill=False, edgecolor="#8a8170", linestyle="--", linewidth=1.0))

    vis = visibility(layout)
    # אלמנטים בלי מיקום: אזור אי-ודאות מקווקו (האזור שהיחסים מאפשרים), לא צורה בנקודה מומצאת
    for el in layout["elements"]:
        if el["position"] != "unspecified":
            continue
        r = uncertainty_box(layout, el)
        ax.add_patch(Rectangle((r.x0, r.y0), r.x1 - r.x0, r.y1 - r.y0, fill=False, hatch="///",
                               edgecolor=(*_rgb(PALETTE[el["class"]]), 0.55), linewidth=0, zorder=0.5))
        ax.text(r.x0 + 0.1, r.y1 - 0.1, f"{el['id']}? {el['phrase']}: location unspecified",
                ha="left", va="top", fontsize=7.5, color=_rgb(PALETTE[el["class"]]), zorder=10,
                bbox=dict(facecolor=_rgb(PALETTE["background"]), edgecolor="none", pad=1.5))

    for z, el in enumerate(drawn_elements(layout), start=1):
        bb, st = el["bbox"], EVIDENCE_STYLE[el["evidence"]]
        if vis[el["id"]]["status"] == "occluded":
            # מוסתר לחלוטין (נמצא מתחת לאלמנט אחר): קו מתאר מנוקד מעל הכול, כ-paradata בלבד
            shape_cls = Ellipse if el["shape"] == "ellipse" else Rectangle
            args = ((bb["x"] + bb["w"] / 2, bb["y"] + bb["h"] / 2), bb["w"], bb["h"]) if el["shape"] == "ellipse" \
                else ((bb["x"], bb["y"]), bb["w"], bb["h"])
            ax.add_patch(shape_cls(*args, fill=False, edgecolor="#5c5446", linestyle=(0, (1, 2)),
                                   linewidth=1.2, zorder=9))
            ax.text(bb["x"] + bb["w"] / 2, bb["y"] + bb["h"] / 2, f"{el['id']}\n(below)", ha="center",
                    va="center", fontsize=7, color="#5c5446", zorder=10)
            continue
        color = _rgb(PALETTE[el["class"]])
        common = dict(facecolor=(*color, st["fill_alpha"]), edgecolor="#2b2118",
                      linestyle=st["linestyle"], linewidth=st["linewidth"], zorder=1 + z * 0.01)
        if el["shape"] == "ellipse":
            patch = Ellipse((bb["x"] + bb["w"] / 2, bb["y"] + bb["h"] / 2), bb["w"], bb["h"], **common)
        else:
            patch = Rectangle((bb["x"], bb["y"]), bb["w"], bb["h"], **common)
        ax.add_patch(patch)
        label = el["id"] + (f"\n{el['locus']}" if el["locus"] else "")
        dark_fill = el["class"] in ("wall", "pit") and el["evidence"] == "attested"
        # מיקום תווית: רצפות בפינה הצפון-מערבית (כדי לא להתנגש באלמנטים שעליהן),
        # אלמנטים קטנים מימין להם, והשאר במרכז.
        if el["class"] == "floor":
            pos, ha, va, color = (bb["x"] + 0.15, bb["y"] + bb["h"] - 0.15), "left", "top", "#2b2118"
        elif max(bb["w"], bb["h"]) < 0.75:  # קטן בשני הממדים (קירות צרים אך ארוכים - לא כאן)
            pos, ha, va, color = (bb["x"] + bb["w"] + 0.1, bb["y"] + bb["h"] / 2), "left", "center", "#2b2118"
        else:
            pos, ha, va = (bb["x"] + bb["w"] / 2, bb["y"] + bb["h"] / 2), "center", "center"
            color = "white" if dark_fill else "#2b2118"
        ax.text(*pos, label, ha=ha, va=va, fontsize=7.5, color=color, zorder=10)
        # חצים קטנים בצלעות שנמשכות מעבר לגבול החפירה
        for side in el["truncated"]:
            cx, cy = bb["x"] + bb["w"] / 2, bb["y"] + bb["h"] / 2
            tip = {"N": (cx, H + 0.25), "S": (cx, -0.25), "E": (W + 0.25, cy), "W": (-0.25, cy)}[side]
            base = {"N": (cx, H), "S": (cx, 0), "E": (W, cy), "W": (0, cy)}[side]
            ax.annotate("", xy=tip, xytext=base, arrowprops=dict(arrowstyle="->", color="#5c5446", lw=1), zorder=11)

    # חץ צפון וקנה מידה
    span = max(W, H)
    ax.annotate("N", xy=(W + span * 0.07, H), xytext=(W + span * 0.07, H - span * 0.12),
                ha="center", fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="#2b2118", lw=1.5))
    ax.plot([0, 1], [-span * 0.07] * 2, color="#2b2118", lw=3, solid_capstyle="butt")
    ax.text(0.5, -span * 0.11, "1 m", ha="center", va="top", fontsize=8)

    class_handles = [Patch(facecolor=_rgb(PALETTE[c]), edgecolor="#2b2118", label=c)
                     for c in ("wall", "floor", "installation", "pit")]
    ev_handles = [Line2D([0], [0], color="#2b2118", linestyle=EVIDENCE_STYLE[e]["linestyle"], lw=1.6, label=e)
                  for e in ("attested", "inferred", "hypothetical")]
    ax.legend(handles=class_handles + ev_handles, loc="upper center", bbox_to_anchor=(0.5, 0.0),
              ncol=7, fontsize=7.5, frameon=False, handlelength=2.2)

    src = layout["source"]
    ax.set_title(f"{src['id']}  |  {src['period']}  |  {layout['area']['label']} ({W:g}x{H:g} m)", fontsize=10)
    ax.set_xlim(-span * 0.1, W + span * 0.14)
    ax.set_ylim(-span * 0.16, H + span * 0.1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="רינדור ArchLayout לאיור, מסכה ותיבות")
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--size", type=int, default=512, help="גודל קנבס המסכה בפיקסלים")
    ap.add_argument("--force", action="store_true", help="לרנדר גם Layout עם שגיאות אימות")
    args = ap.parse_args()

    for path in args.paths:
        layout = load_layout(path)
        errs = errors_only(validate(layout))
        if errs and not args.force:
            print(f"{path}: דילוג - ה-Layout לא תקין ({len(errs)} שגיאות). הרץ validate_layout.py לפרטים.")
            continue
        stem = path.parent / path.name.removesuffix(".json")
        render_schematic(layout, Path(f"{stem}.schematic.png"))
        render_seg(layout, args.size).save(f"{stem}.seg.png")
        vis = visibility(layout, args.size)
        to_px, _ = image_transform(layout, args.size)
        uncertain = []
        for el in layout["elements"]:
            if el["position"] == "unspecified":
                r = uncertainty_box(layout, el)
                (px0, py0), (px1, py1) = to_px(r.x0, r.y1), to_px(r.x1, r.y0)
                uncertain.append({"id": el["id"], "class": el["class"], "phrase": el["phrase"],
                                  "region": [round(v / args.size, 4) for v in (px0, py0, px1, py1)]})
        Path(f"{stem}.boxes.json").write_text(json.dumps({
            "caption": global_caption(layout),
            "canvas_px": args.size,
            "palette": {k: list(v) for k, v in PALETTE.items()},
            # רק מה שנראה במבט-על ובמקום ידוע מקבל תיבה להתניה
            "boxes": image_boxes(layout, args.size, {i for i, v in vis.items() if v["status"] == "visible"}),
            "occluded": [i for i, v in vis.items() if v["status"] == "occluded"],
            "uncertain": uncertain,
            "visibility": vis,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{path}: נוצרו schematic.png, seg.png, boxes.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
