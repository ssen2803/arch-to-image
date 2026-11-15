"""מפת תקופה וחתך אנכי מתוך Layout עם תיארוך וגבהים (סכמה 1.2).

הפריסה הרגילה מציגה את כל מה שנחשף, מכל התקופות יחד. כאן מציגים את האתר בתקופה אחת:

1. מפת תקופה: האלמנטים שהתיארוך שלהם חופף לתקופה המבוקשת. הנראות מחושבת רק ביניהם,
   כך שרצפה שנאטמה מאוחר יותר נראית במלואה בתקופה שלה. אלמנטים מוקדמים יותר מוצגים כקו מתאר
   אפור (earlier), כי לא ידוע אם היו עדיין גלויים. אלמנטים בלי תיארוך מוצגים באותו אופן (undated).
   אלמנטים מאוחרים יותר לא מוצגים.
2. חתך אנכי: קו לאורך x או y קבוע. כל אלמנט שהקו חוצה ושיש לו גובה מצויר כמלבן בין הבסיס לראש.
   בסיס לא ידוע מצויר כרצועה דקה עם קו תחתון מקווקו. אלמנטים בלי גובה לא מצוירים, ונרשמים בכותרת.

שימוש:
    python src/render_period.py examples/ex08_two_phases/layout.gold.json --section x 3.0
    python src/render_period.py layout.json --period "Iron Age II" -800 -701 --section y 4.5
"""

from __future__ import annotations

import argparse
import copy
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layout_core import PALETTE, Box, load_layout, stratigraphic_order  # noqa: E402
from render_layout import EVIDENCE_STYLE, _rgb, render_schematic, visibility  # noqa: E402


def dating_interval(el: dict) -> tuple[int, int] | None:
    d = el.get("dating")
    if not d or d.get("start_year") is None or d.get("end_year") is None:
        return None
    return d["start_year"], d["end_year"]


def periods_in(layout: dict) -> list[tuple[str, int, int]]:
    """התקופות שמופיעות בתיארוכי האלמנטים, מהמוקדמת למאוחרת."""
    found = {(el["dating"]["period"], *iv) for el in layout["elements"] if (iv := dating_interval(el))}
    return sorted(found, key=lambda p: (p[1], p[2]))


def slice_layout(layout: dict, start: int, end: int) -> tuple[dict, list[dict]]:
    """(Layout של התקופה, אלמנטי הקשר). חפיפה בין טווחים: הטווח של האלמנט נוגע בטווח המבוקש."""
    current, context = [], []
    for el in layout["elements"]:
        iv = dating_interval(el)
        if iv is None:
            context.append({**el, "_context": "undated"})
        elif iv[1] < start:
            context.append({**el, "_context": "earlier"})
        elif iv[0] <= end:
            current.append(el)
    ids = {e["id"] for e in current}
    sliced = copy.deepcopy(layout)
    sliced["elements"] = copy.deepcopy(current)
    sliced["relations"] = [r for r in layout["relations"] if r["subject"] in ids and r["object"] in ids]
    return sliced, context


def _year(y: int) -> str:
    return f"{-y} BCE" if y < 0 else f"{y} CE"


def render_period_map(layout: dict, period: str, start: int, end: int, out: Path) -> dict:
    sliced, context = slice_layout(layout, start, end)
    if not sliced["elements"]:
        raise ValueError(f"אין אלמנטים מתוארכים לתקופה {period}")
    title = f"{layout['source']['id']}  |  {period} ({_year(start)} - {_year(end)})  |  {layout['area']['label']}"
    render_schematic(sliced, out, context=context, title=title)
    return {"current": [e["id"] for e in sliced["elements"]],
            "earlier": [e["id"] for e in context if e["_context"] == "earlier"],
            "undated": [e["id"] for e in context if e["_context"] == "undated"],
            "visibility": visibility(sliced)}


def section_elements(layout: dict, axis: str, coord: float) -> tuple[list[dict], list[str]]:
    """האלמנטים שהקו x=coord (או y=coord) חוצה, בסדר סטרטיגרפי מלמטה למעלה, ומי מהם בלי גובה."""
    order = stratigraphic_order(layout) or layout["elements"]
    hit, no_elev = [], []
    for el in order:
        b = Box.from_bbox(el["bbox"])
        lo, hi = (b.x0, b.x1) if axis == "x" else (b.y0, b.y1)
        if not (lo <= coord <= hi):
            continue
        elev = el.get("elevation")
        if not elev or elev.get("top_m") is None:
            no_elev.append(el["id"])
            continue
        hit.append(el)
    return hit, no_elev


def render_section(layout: dict, axis: str, coord: float, out: Path) -> dict:
    els, no_elev = section_elements(layout, axis, coord)
    if not els:
        raise ValueError(f"הקו {axis}={coord} לא חוצה אף אלמנט עם גובה")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bg = _rgb(PALETTE["background"])
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    along_max = layout["area"]["height_m"] if axis == "x" else layout["area"]["width_m"]
    tops = [e["elevation"]["top_m"] for e in els]
    bottoms = [e["elevation"]["bottom_m"] for e in els if e["elevation"].get("bottom_m") is not None]
    ymin, ymax = min(bottoms + tops) - 0.3, max(tops) + 0.3
    for z, el in enumerate(els, start=1):          # מלמטה למעלה: אלמנט מאוחר (בור חותך) מצויר מעל
        b = Box.from_bbox(el["bbox"])
        a0, a1 = (b.y0, b.y1) if axis == "x" else (b.x0, b.x1)
        top, bottom = el["elevation"]["top_m"], el["elevation"].get("bottom_m")
        st = EVIDENCE_STYLE[el["evidence"]]
        color = _rgb(PALETTE[el["class"]])
        h = (top - bottom) if bottom is not None else 0.06
        ax.add_patch(Rectangle((a0, top - h), a1 - a0, h, facecolor=(*color, st["fill_alpha"]), edgecolor="#2b2118",
                               linestyle=st["linestyle"], linewidth=st["linewidth"], zorder=1 + z * 0.01))
        if bottom is None:
            ax.plot([a0, a1], [top - 0.25] * 2, color="#7d7466", linestyle=(0, (2, 2)), lw=0.9, zorder=1)
        period = (el.get("dating") or {}).get("period", "undated")
        label = f"{el['id']} {el.get('locus') or ''}\n{period}"
        if h >= 0.35:      # אלמנט עבה (קיר, בור): תווית בתוכו
            ax.text((a0 + a1) / 2, top - h / 2, label, ha="center", va="center", fontsize=7, color="white", zorder=10)
        else:              # רצועה דקה (רצפה): תווית מעל הקצה השמאלי שלה
            ax.text(a0 + 0.05, top + 0.03, label, ha="left", va="bottom", fontsize=7, color="#2b2118", zorder=10)
    ax.set_xlim(-0.3, along_max + 0.3)
    ax.set_ylim(ymin, ymax + 0.35)
    datum = els[0]["elevation"]["datum"]
    ax.set_ylabel("elevation (m above sea level)" if datum == "asl" else "elevation (m, local datum)")
    ax.set_xlabel(("south -> north" if axis == "x" else "west -> east") + " (m in area frame)")
    note = f"   |   no elevation: {', '.join(no_elev)}" if no_elev else ""
    ax.set_title(f"{layout['source']['id']}  |  section along {axis} = {coord:g} m{note}", fontsize=9.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], color="#7d7466", linestyle=(0, (2, 2)), lw=0.9,
                              label="bottom not stated (thickness unknown)")],
              loc="upper right", fontsize=7, frameon=False)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"elements": [e["id"] for e in els], "no_elevation": no_elev}


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="מפת תקופה וחתך אנכי")
    ap.add_argument("path", type=Path)
    ap.add_argument("--period", nargs=3, action="append", metavar=("NAME", "START", "END"),
                    help="תקופה וטווח שנים (שלילי = לפנה\"ס). ברירת מחדל: כל התקופות שבתיארוכים")
    ap.add_argument("--section", nargs=2, metavar=("AXIS", "COORD"), help="x או y, וקואורדינטה במטרים")
    args = ap.parse_args()

    layout = load_layout(args.path)
    stem = args.path.with_suffix("")
    periods = [(n, int(s), int(e)) for n, s, e in args.period] if args.period else periods_in(layout)
    for name, start, end in periods:
        out = Path(f"{stem}.period_{_slug(name)}.png")
        info = render_period_map(layout, name, start, end, out)
        print(f"{out.name}: בתקופה {info['current']}, מוקדמים {info['earlier']}, לא מתוארכים {info['undated']}")
    if args.section:
        axis, coord = args.section[0], float(args.section[1])
        out = Path(f"{stem}.section_{axis}{coord:g}.png")
        info = render_section(layout, axis, coord, out)
        print(f"{out.name}: {info['elements']}" + (f", בלי גובה: {info['no_elevation']}" if info["no_elevation"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
