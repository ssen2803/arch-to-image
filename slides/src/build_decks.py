"""Builds the self-contained HTML decks from the templates in this folder.

Results on a slide are read from result files in the repository (results/, 03_controlled_generation/outputs/,
02_layout_pipeline/examples/), so slides, README and result files agree. Hardware figures (memory, time) are typed by hand.

    ..\\..\\.venv\\Scripts\\python build_decks.py              # proposal.html + midterm.html
    ..\\..\\.venv\\Scripts\\python build_decks.py midterm      # one deck

Template syntax: {{ python expression }}. The expression sees the helpers and the dictionaries defined in context().
"""

from __future__ import annotations

import ast
import base64
import csv
import html
import io
import itertools
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "02_layout_pipeline" / "src"))
from layout_core import image_transform, load_layout, uncertainty_box  # noqa: E402
from render_layout import visibility  # noqa: E402
from compare_layouts import compare  # noqa: E402

EX = ROOT / "02_layout_pipeline" / "examples"
GEN = ROOT / "03_controlled_generation" / "outputs"
INK, INK2, LINE = "#15222A", "#465761", "#B5C4BF"
RED, BLUE, OCHRE, GREY = "#C1272D", "#1F62B8", "#A8690A", "#8E9AA1"
FILL = {"wall": "#5C4033", "floor": "#E2C58D", "installation": "#C44E38", "pit": "#3A5A78"}
esc = html.escape


# ------------------------------------------------------------------ images

def img(path, w=None, h=None, crop=None, q=86, alt="", style=""):
    """<img> with the file inlined as JPEG. crop = (x0, y0, x1, y1) as fractions."""
    im = Image.open(ROOT / path).convert("RGB")
    if crop:
        im = im.crop((round(crop[0] * im.width), round(crop[1] * im.height), round(crop[2] * im.width), round(crop[3] * im.height)))
    if w:
        im = im.resize((w, h or round(w * im.height / im.width)), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, "JPEG", quality=q, optimize=True)
    return f'<img src="data:image/jpeg;base64,{base64.b64encode(b.getvalue()).decode()}" alt="{esc(alt)}" style="{style}">'


def png(path, w=None, alt="", style=""):
    im = Image.open(ROOT / path)
    if w:
        im = im.resize((w, round(w * im.height / im.width)), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, "PNG", optimize=True)
    return f'<img src="data:image/png;base64,{base64.b64encode(b.getvalue()).decode()}" alt="{esc(alt)}" style="{style}">'


def label_img(path):
    """Class mask in the project palette."""
    pal = np.array([(240, 234, 222), (214, 178, 112), (92, 64, 51), (196, 78, 56), (58, 90, 120)], np.uint8)
    im = Image.fromarray(pal[np.array(Image.open(ROOT / path))]).resize((300, 300), Image.NEAREST)
    b = io.BytesIO()
    im.save(b, "PNG", optimize=True)
    return f'<img src="data:image/png;base64,{base64.b64encode(b.getvalue()).decode()}" alt="Label mask">'


def wall_outline(tile, w=420):
    """Real tile with the published wall polygons drawn on it."""
    src = ROOT / "04_cv_experiment/data/real/uziel_extracted/annotated_tiles/wall_tiles"
    im = Image.open(src / f"{tile}.jpg").convert("RGB")
    d = ImageDraw.Draw(im)
    for sh in json.loads((src / f"{tile}.json").read_text(encoding="utf-8"))["shapes"]:
        pts = [tuple(q) for q in sh["points"]]
        d.line(pts + [pts[0]], fill=(255, 205, 40), width=7)
    im = im.resize((w, w), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, "JPEG", quality=87)
    return f'<img src="data:image/jpeg;base64,{base64.b64encode(b.getvalue()).decode()}" alt="Drone tile with wall outlines">'


def overlay_mask(img_path, gt_path, pred_path, w=420):
    """Photo with TP (green), FP (red) and FN (yellow) wall pixels. Returns <img>."""
    im = Image.open(ROOT / img_path).convert("RGB")
    a = np.array(im).astype(np.float32)
    gt = np.array(Image.open(ROOT / gt_path))
    pr = np.array(Image.open(ROOT / pred_path)) > 127
    g, ok = gt == 2, gt != 255
    for m, c in (((g & pr & ok), (30, 190, 110)), ((~g & pr & ok), (214, 40, 40)), ((g & ~pr & ok), (255, 205, 40))):
        a[m] = 0.42 * a[m] + 0.58 * np.array(c, np.float32)
    out = Image.fromarray(a.astype(np.uint8)).resize((w, w), Image.LANCZOS)
    b = io.BytesIO()
    out.save(b, "JPEG", quality=87)
    return f'<img src="data:image/jpeg;base64,{base64.b64encode(b.getvalue()).decode()}" alt="Wall prediction over the photo">'


# ------------------------------------------------------------------ plans drawn from a layout file

def plan(layout, s=90, pad=46, labels=True, cls="", fs=20, frame=True, scale=True, photo_frame=False, loc=True, notes=True):
    """SVG plan of a layout (metres -> s px). Evidence level = line style. Unknown position = hatched zone.
    photo_frame=True draws in the 512-canvas frame used by the generator, so it registers with a generated image."""
    L = load_layout(EX / layout / "layout.gold.json") if isinstance(layout, str) else layout
    W, H = L["area"]["width_m"], L["area"]["height_m"]
    if photo_frame:
        size = s
        to_px, k = image_transform(L, size)
        vb = f"0 0 {size} {size}"
        tx = lambda x: to_px(x, 0)[0]                       # noqa: E731
        ty = lambda y: to_px(0, y)[1]                       # noqa: E731
        wpx = hpx = size
    else:
        k = s
        wpx, hpx = W * s + 2 * pad, H * s + 2 * pad
        vb = f"0 0 {wpx:.0f} {hpx:.0f}"
        tx = lambda x: pad + x * s                          # noqa: E731
        ty = lambda y: pad + (H - y) * s                    # noqa: E731
    vis = visibility(L)
    uid = f"p{abs(hash((str(layout) if isinstance(layout, str) else id(layout), s, cls))) % 99999}"
    o = [f'<svg viewBox="{vb}" width="{wpx:.0f}" height="{hpx:.0f}" class="{cls}" role="img" aria-label="Plan drawn from the layout">']
    o.append(f'<defs><pattern id="{uid}h" width="14" height="14" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
             f'<rect width="14" height="14" fill="none"/><line x1="0" y1="0" x2="0" y2="14" stroke="{FILL["pit"]}" stroke-width="3" opacity=".55"/></pattern>'
             f'<pattern id="{uid}d" width="16" height="16" patternUnits="userSpaceOnUse"><circle cx="4" cy="4" r="1.6" fill="#9C7B3A"/><circle cx="12" cy="12" r="1.6" fill="#9C7B3A"/></pattern></defs>')
    if frame:
        o.append(f'<rect x="{tx(0):.1f}" y="{ty(H):.1f}" width="{W * k:.1f}" height="{H * k:.1f}" fill="none" stroke="{INK2}" stroke-width="2" stroke-dasharray="12 7"/>')
    order = {"floor": 0, "pit": 1, "wall": 2, "installation": 3}
    els = sorted(L["elements"], key=lambda e: order[e["class"]])
    # a buried element is drawn last, as a dotted outline
    els = [e for e in els if vis[e["id"]]["status"] != "occluded"] + [e for e in els if vis[e["id"]]["status"] == "occluded"]
    for i, e in enumerate(els):
        st, bb, c = vis[e["id"]]["status"], e["bbox"], e["class"]
        x, y, w, h = tx(bb["x"]), ty(bb["y"] + bb["h"]), bb["w"] * k, bb["h"] * k
        cx, cy = x + w / 2, y + h / 2
        fo = {"attested": 1, "inferred": .5, "hypothetical": .10}[e["evidence"]]
        da = {"attested": "", "inferred": "16 9", "hypothetical": "3 8"}[e["evidence"]]
        g = [f'<g class="el" data-k="{e["id"]}" style="--i:{i}">']
        dash = f' stroke-dasharray="{da}"' if da else ""
        if st == "unspecified":
            b = uncertainty_box(L, e)
            X, Y, BW, BH = tx(b.x0), ty(b.y1), (b.x1 - b.x0) * k, (b.y1 - b.y0) * k
            g.append(f'<rect class="shape" x="{X:.1f}" y="{Y:.1f}" width="{BW:.1f}" height="{BH:.1f}" fill="url(#{uid}h)" stroke="{FILL["pit"]}" stroke-width="2.5" stroke-dasharray="10 8"/>')
            if labels:
                g.append(f'<rect x="{X + BW / 2 - 150:.1f}" y="{Y + BH / 2 - 40:.1f}" width="300" height="80" fill="#fff" opacity=".92"/>'
                         f'<text class="lbl" x="{X + BW / 2:.1f}" y="{Y + BH / 2 - 6:.1f}" font-size="{fs + 3}" text-anchor="middle" font-weight="600">{e["id"]}: pit, 1 m</text>'
                         f'<text class="lbl" x="{X + BW / 2:.1f}" y="{Y + BH / 2 + 24:.1f}" font-size="{fs}" text-anchor="middle">somewhere in this zone</text>')
            g.append("</g>")
            o += g
            continue
        if st == "occluded":
            g.append(f'<ellipse class="shape" cx="{cx:.1f}" cy="{cy:.1f}" rx="{w / 2:.1f}" ry="{h / 2:.1f}" fill="none" stroke="{FILL["pit"]}" stroke-width="3.5" stroke-dasharray="4 8"/>')
            if labels:
                g.append(f'<text class="lbl" x="{cx:.1f}" y="{cy - 2:.1f}" font-size="{fs}" text-anchor="middle" font-weight="600">{e["id"]}</text>'
                         f'<text class="lbl" x="{cx:.1f}" y="{cy + fs + 1:.1f}" font-size="{fs - 3}" text-anchor="middle">sealed below</text>')
            g.append("</g>")
            o += g
            continue
        shape = (f'<ellipse class="shape" cx="{cx:.1f}" cy="{cy:.1f}" rx="{w / 2:.1f}" ry="{h / 2:.1f}"' if e["shape"] == "ellipse"
                 else f'<rect class="shape" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}"')
        g.append(f'{shape} fill="{FILL[c]}" fill-opacity="{fo}" stroke="{INK}" stroke-width="2.5"{dash}/>')
        if c == "floor" and e["evidence"] == "attested":
            g.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="url(#{uid}d)" opacity=".5" pointer-events="none"/>')
        if c == "pit":                                     # hachures pointing into the cut
            for a in np.linspace(0, 2 * np.pi, 18, endpoint=False):
                r1, r2 = .98, .78
                g.append(f'<line x1="{cx + np.cos(a) * w / 2 * r1:.1f}" y1="{cy + np.sin(a) * h / 2 * r1:.1f}" x2="{cx + np.cos(a) * w / 2 * r2:.1f}" y2="{cy + np.sin(a) * h / 2 * r2:.1f}" stroke="#fff" stroke-width="2"/>')
        for side in e.get("truncated", []):
            ax, ay, dx, dy = {"N": (cx, y, 0, -1), "S": (cx, y + h, 0, 1), "E": (x + w, cy, 1, 0), "W": (x, cy, -1, 0)}[side]
            g.append(f'<path d="M{ax + dx * 6 - dy * 8:.1f} {ay + dy * 6 - dx * 8:.1f} L{ax + dx * 18:.1f} {ay + dy * 18:.1f} L{ax + dx * 6 + dy * 8:.1f} {ay + dy * 6 + dx * 8:.1f}" fill="none" stroke="{INK}" stroke-width="2.5"/>')
        if labels:
            dark = c in ("wall", "pit", "installation") and e["evidence"] != "hypothetical"
            col = "#fff" if dark else INK
            txt = e["id"] + (f' ({e["locus"]})' if loc and e.get("locus") and c != "installation" else "")
            if w < 130 and h > w * 1.6:
                g.append(f'<text class="lbl" transform="translate({cx:.1f} {cy:.1f}) rotate(-90)" y="{fs * .35:.1f}" font-size="{fs}" text-anchor="middle" style="fill:{col}" font-weight="600">{txt}</text>')
            elif c == "installation" and w < 62:
                g.append(f'<text class="lbl" x="{cx:.1f}" y="{cy + fs * .35:.1f}" font-size="{fs - 2}" text-anchor="middle" style="fill:{col}" font-weight="600">{e["id"]}</text>')
            else:
                g.append(f'<text class="lbl" x="{cx:.1f}" y="{cy + fs * .35:.1f}" font-size="{fs}" text-anchor="middle" style="fill:{col}" font-weight="600">{txt}</text>')
        g.append("</g>")
        o += g
    if scale and not photo_frame:
        y0 = hpx - 16
        o.append(f'<g><rect x="{pad}" y="{y0 - 8}" width="{k}" height="8" fill="{INK}"/><rect x="{pad + k}" y="{y0 - 8}" width="{k}" height="8" fill="#fff" stroke="{INK}" stroke-width="1.5"/>'
                 f'<text x="{pad + 2 * k + 10}" y="{y0}" font-size="{fs}">2 m</text>'
                 f'<path d="M{wpx - pad + 16} {pad + 34} v-44 m-9 14 l9 -14 l9 14" fill="none" stroke="{INK}" stroke-width="3"/><text x="{wpx - pad + 16}" y="{pad + 60}" font-size="{fs}" text-anchor="middle">N</text></g>')
    o.append("</svg>")
    return "".join(o)


def plan_compare(ex, s=80, pad=40, fs=18, pred="layout.pred.json"):
    """Reference layout (blue outline) and LLM layout (red outline) on one plan, matched by their quotes."""
    g, q = load_layout(EX / ex / "layout.gold.json"), load_layout(EX / ex / pred)
    W, H = g["area"]["width_m"], g["area"]["height_m"]
    w, h = W * s + 2 * pad, H * s + 2 * pad
    o = [f'<svg viewBox="0 0 {w:.0f} {h:.0f}" width="{w:.0f}" height="{h:.0f}" role="img" aria-label="Reference and extracted layout on one plan">'
         f'<rect x="{pad}" y="{pad}" width="{W * s}" height="{H * s}" fill="none" stroke="{INK2}" stroke-width="2" stroke-dasharray="12 7"/>']
    for L, c, wd, dash in ((g, BLUE, 4, ""), (q, RED, 3.5, ' stroke-dasharray="11 6"')):
        for e in L["elements"]:
            b = e["bbox"]
            x, y, ww, hh = pad + b["x"] * s, pad + (H - b["y"] - b["h"]) * s, b["w"] * s, b["h"] * s
            fill = FILL[e["class"]] if c == BLUE else "none"
            if e["shape"] == "ellipse":
                o.append(f'<ellipse cx="{x + ww / 2:.1f}" cy="{y + hh / 2:.1f}" rx="{ww / 2:.1f}" ry="{hh / 2:.1f}" fill="{fill}" fill-opacity=".16" stroke="{c}" stroke-width="{wd}"{dash}/>')
            else:
                o.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{ww:.1f}" height="{hh:.1f}" fill="{fill}" fill-opacity=".16" stroke="{c}" stroke-width="{wd}"{dash}/>')
    o.append("</svg>")
    return "".join(o)


# ------------------------------------------------------------------ charts (inline SVG)

def hbar(rows, w=760, rowh=54, label_w=250, xmax=1.0, fmt="{:.2f}", ticks=None, title="", fs=22, thick=22):
    """rows: dicts with label, v, color and optionally lo/hi (95% CI), tip, bold."""
    top = 44 if title else 10
    h = top + rowh * len(rows) + 40
    x0, x1 = label_w, w - 96
    X = lambda v: x0 + (x1 - x0) * v / xmax                 # noqa: E731
    o = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{esc(title)}">']
    if title:
        o.append(f'<text x="0" y="24" font-size="{fs + 2}" font-weight="600">{esc(title)}</text>')
    for t in ticks or [0, xmax / 2, xmax]:
        o.append(f'<line class="gl" x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{top}" y2="{h - 36}"/><text class="ax" x="{X(t):.1f}" y="{h - 12}" font-size="{fs - 3}" text-anchor="middle">{t:g}</text>')
    for i, r in enumerate(rows):
        cy = top + rowh * (i + .5)
        v = r["v"]
        bw = max(2, X(max(v, 0)) - x0)
        tip = r.get("tip") or f'{r["label"]}: {fmt.format(v)}' + (f'\n95% CI {fmt.format(r["lo"])} to {fmt.format(r["hi"])}' if "lo" in r else "")
        wt = ' font-weight="600"' if r.get("bold") else ""
        o.append(f'<text x="{x0 - 14}" y="{cy + fs * .35:.1f}" font-size="{fs}" text-anchor="end"{wt}>{esc(r["label"])}</text>')
        o.append(f'<path data-tip="{esc(tip)}" d="M{x0} {cy - thick / 2:.1f} h{bw - 4:.1f} a4 4 0 0 1 4 4 v{thick - 8} a4 4 0 0 1 -4 4 h{-(bw - 4):.1f} z" fill="{r["color"]}"/>')
        end = X(max(v, 0))
        if "lo" in r:
            o.append(f'<line x1="{X(r["lo"]):.1f}" x2="{X(r["hi"]):.1f}" y1="{cy:.1f}" y2="{cy:.1f}" stroke="{INK}" stroke-width="2.5"/>'
                     f'<line x1="{X(r["lo"]):.1f}" x2="{X(r["lo"]):.1f}" y1="{cy - 7:.1f}" y2="{cy + 7:.1f}" stroke="{INK}" stroke-width="2.5"/>'
                     f'<line x1="{X(r["hi"]):.1f}" x2="{X(r["hi"]):.1f}" y1="{cy - 7:.1f}" y2="{cy + 7:.1f}" stroke="{INK}" stroke-width="2.5"/>')
            end = max(end, X(r["hi"]))
        o.append(f'<text x="{end + 12:.1f}" y="{cy + fs * .35:.1f}" font-size="{fs}"{wt}>{fmt.format(v)}</text>')
    o.append(f'<line class="bl" x1="{x0}" x2="{x0}" y1="{top}" y2="{h - 36}"/></svg>')
    return "".join(o)


def hist2(a, b, la, lb, w=560, h=300, lo=0, hi=1, bins=24, title="", unit="", fs=20, ca=RED, cb=BLUE, ticks=None, tickfmt="{:g}"):
    """Two normalised histograms on one axis (share of images per bin), with a legend and the medians marked."""
    ha, e = np.histogram(np.clip(a, lo, hi), bins=bins, range=(lo, hi))
    hb, _ = np.histogram(np.clip(b, lo, hi), bins=bins, range=(lo, hi))
    ha, hb = ha / max(1, ha.sum()), hb / max(1, hb.sum())
    top, base, x0, x1 = 78, h - 40, 24, w - 30
    m = max(ha.max(), hb.max())
    X = lambda v: x0 + (x1 - x0) * (v - lo) / (hi - lo)     # noqa: E731
    Y = lambda v: base - (base - top) * v / m               # noqa: E731
    o = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{esc(title)}"><text x="0" y="24" font-size="{fs + 3}" font-weight="600">{esc(title)}</text>']
    lx = 0
    for c, n, d in ((ca, la, a), (cb, lb, b)):
        t = f"{n}, median {tickfmt.format(np.median(d))}{unit}"
        o.append(f'<rect x="{lx}" y="42" width="18" height="18" fill="{c}" fill-opacity=".75"/><text x="{lx + 26}" y="58" font-size="{fs - 1}">{esc(t)}</text>')
        lx += 36 + len(t) * (fs - 1) * .50
    for hh, c, n in ((ha, ca, la), (hb, cb, lb)):
        for i, v in enumerate(hh):
            if v > 0:
                o.append(f'<rect data-tip="{esc(n)}: {v:.0%} of images between {tickfmt.format(e[i])} and {tickfmt.format(e[i + 1])}" x="{X(e[i]) + 1:.1f}" y="{Y(v):.1f}" width="{X(e[i + 1]) - X(e[i]) - 2:.1f}" height="{base - Y(v):.1f}" fill="{c}" fill-opacity=".62"/>')
    for d, c in ((a, ca), (b, cb)):
        mx = X(min(max(float(np.median(d)), lo), hi))
        o.append(f'<line x1="{mx:.1f}" x2="{mx:.1f}" y1="{top - 8}" y2="{base}" stroke="{c}" stroke-width="3"/>')
    for t in ticks or np.linspace(lo, hi, 5):
        o.append(f'<text class="ax" x="{X(t):.1f}" y="{h - 12}" font-size="{fs - 2}" text-anchor="middle">{tickfmt.format(t)}</text>')
    o.append(f'<line class="bl" x1="{x0}" x2="{x1}" y1="{base}" y2="{base}"/></svg>')
    return "".join(o)


def hist1(a, w=560, h=280, lo=0, hi=1, bins=25, title="", color=BLUE, marks=(), fs=20, ticks=None):
    ha, e = np.histogram(np.clip(a, lo, hi), bins=bins, range=(lo, hi))
    top, base, x0, x1 = 64, h - 40, 24, w - 30
    X = lambda v: x0 + (x1 - x0) * (v - lo) / (hi - lo)     # noqa: E731
    Y = lambda v: base - (base - top) * v / ha.max()        # noqa: E731
    o = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{esc(title)}"><text x="0" y="24" font-size="{fs + 3}" font-weight="600">{esc(title)}</text>']
    for i, v in enumerate(ha):
        if v:
            o.append(f'<rect data-tip="{v} images between {e[i]:.2f} and {e[i + 1]:.2f}" x="{X(e[i]) + 1:.1f}" y="{Y(v):.1f}" width="{X(e[i + 1]) - X(e[i]) - 2:.1f}" height="{base - Y(v):.1f}" fill="{color}"/>')
    for v, t in marks:
        o.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{top - 18}" y2="{base}" stroke="{RED}" stroke-width="3"/><text x="{X(v) + 8:.1f}" y="{top - 4}" font-size="{fs - 1}">{esc(t)}</text>')
    for t in ticks or np.linspace(lo, hi, 6):
        o.append(f'<text class="ax" x="{X(t):.1f}" y="{h - 12}" font-size="{fs - 2}" text-anchor="middle">{t:g}</text>')
    o.append(f'<line class="bl" x1="{x0}" x2="{x1}" y1="{base}" y2="{base}"/></svg>')
    return "".join(o)


def strips(groups, w=620, rowh=92, lo=0, hi=.7, title="", fs=21, label_w=0):
    """Quartile strips: groups = [(label, values, color)]. Box = middle half, line = median."""
    h = 48 + rowh * len(groups) + 36
    x0, x1 = label_w + 22, w - 70
    X = lambda v: x0 + (x1 - x0) * (v - lo) / (hi - lo)     # noqa: E731
    o = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{esc(title)}"><text x="0" y="24" font-size="{fs + 2}" font-weight="600">{esc(title)}</text>']
    for t in np.arange(lo, hi + .001, .2):
        o.append(f'<line class="gl" x1="{X(t):.1f}" x2="{X(t):.1f}" y1="44" y2="{h - 34}"/><text class="ax" x="{X(t):.1f}" y="{h - 10}" font-size="{fs - 3}" text-anchor="middle">{t:.1f}</text>')
    for i, (lab, v, c) in enumerate(groups):
        y = 48 + rowh * i
        p10, q1, m, q3, p90 = np.percentile(v, [10, 25, 50, 75, 90])
        o.append(f'<text x="0" y="{y + 24}" font-size="{fs}">{esc(lab)}</text>')
        o.append(f'<line x1="{X(max(lo, p10)):.1f}" x2="{X(min(hi, p90)):.1f}" y1="{y + 54}" y2="{y + 54}" stroke="{c}" stroke-width="3"/>'
                 f'<rect data-tip="{esc(lab)}\nmedian {m:.2f}, middle half {q1:.2f} to {q3:.2f}" x="{X(q1):.1f}" y="{y + 42}" width="{X(q3) - X(q1):.1f}" height="24" fill="{c}" rx="3"/>'
                 f'<line x1="{X(m):.1f}" x2="{X(m):.1f}" y1="{y + 38}" y2="{y + 70}" stroke="#fff" stroke-width="4"/>'
                 f'<text x="{X(min(hi, p90)) + 10:.1f}" y="{y + 61}" font-size="{fs}" font-weight="600">{m:.2f}</text>')
    o.append("</svg>")
    return "".join(o)


def share_bar(parts, w=1000, h=96, fs=21):
    """One segmented bar with a legend row above it: parts = [(label, share, color)]."""
    o = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="Pixel share per class">']
    gap, lx = 3, 0.0
    for lab, v, c in parts:
        t = f"{lab} {v:.1%}"
        o.append(f'<rect x="{lx:.1f}" y="10" width="18" height="18" fill="{c}" stroke="{INK}" stroke-width="1"/><text x="{lx + 27:.1f}" y="27" font-size="{fs}">{esc(t)}</text>')
        lx += 27 + len(t) * fs * .54 + 46
    x = 0.0
    for lab, v, c in parts:
        bw = v * (w - gap * (len(parts) - 1))
        o.append(f'<rect data-tip="{esc(lab)}: {v:.1%} of pixels" x="{x:.1f}" y="48" width="{max(bw, 2):.1f}" height="40" fill="{c}" stroke="{INK}" stroke-width="1"/>')
        x += bw + gap
    o.append("</svg>")
    return "".join(o)


def heat(m, rows, cols, w=560, title="", fs=19):
    """Row-normalised confusion matrix, one hue."""
    m = np.array(m, float)
    n = m / np.maximum(m.sum(1, keepdims=True), 1)
    cw, ch, lx, ty = (w - 150) / len(cols), 50, 150, 88
    h = ty + ch * len(rows) + 8
    o = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{esc(title)}"><text x="0" y="24" font-size="{fs + 4}" font-weight="600">{esc(title)}</text><text class="ax" x="{lx}" y="52" font-size="{fs - 1}">predicted as</text>']
    for j, c in enumerate(cols):
        o.append(f'<text x="{lx + cw * (j + .5):.1f}" y="{ty - 10}" font-size="{fs - 2}" text-anchor="middle">{esc(c)}</text>')
    for i, r in enumerate(rows):
        o.append(f'<text x="{lx - 12}" y="{ty + ch * (i + .5) + 6:.1f}" font-size="{fs}" text-anchor="end">{esc(r)}</text>')
        for j in range(len(cols)):
            v = n[i, j]
            o.append(f'<rect data-tip="true {esc(r)}, predicted {esc(cols[j])}: {v:.0%}" x="{lx + cw * j + 1:.1f}" y="{ty + ch * i + 1:.1f}" width="{cw - 2:.1f}" height="{ch - 2}" fill="{BLUE}" fill-opacity="{.06 + .94 * v:.3f}"/>')
            o.append(f'<text x="{lx + cw * (j + .5):.1f}" y="{ty + ch * (i + .5) + 7:.1f}" font-size="{fs}" text-anchor="middle" style="fill:{"#fff" if v > .5 else INK}">{v:.0%}</text>')
    o.append("</svg>")
    return "".join(o)


def dotmatrix(rows, cols, w=1000, rowh=62, label_w=430, fs=23):
    """rows: (label, [1 | .5 | 0 ...], highlight). Full, half and empty dots."""
    top = 92
    h = top + rowh * len(rows) + 6
    cw = (w - label_w) / len(cols)
    o = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="Which capabilities each prior work covers">']
    for j, c in enumerate(cols):
        for li, part in enumerate(c.split("|")):
            o.append(f'<text x="{label_w + cw * (j + .5):.1f}" y="{30 + li * 26}" font-size="{fs - 2}" text-anchor="middle" font-weight="600">{esc(part)}</text>')
    o.append(f'<line class="bl" x1="0" x2="{w}" y1="{top - 14}" y2="{top - 14}"/>')
    for i, (lab, vals, hl) in enumerate(rows):
        y = top + rowh * i
        if hl:
            o.append(f'<rect x="0" y="{y - 10}" width="{w}" height="{rowh - 4}" fill="#FBEDEC"/><rect x="0" y="{y - 10}" width="6" height="{rowh - 4}" fill="{RED}"/>')
        o.append(f'<text x="18" y="{y + rowh / 2 - 4:.1f}" font-size="{fs}"{" font-weight=" + chr(34) + "600" + chr(34) if hl else ""}>{esc(lab)}</text>')
        for j, v in enumerate(vals):
            cx, cy, r = label_w + cw * (j + .5), y + rowh / 2 - 12, 13
            col = RED if hl else INK
            o.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="none" stroke="{col}" stroke-width="2.5"/>')
            if v == 1:
                o.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{col}"/>')
            elif v:
                o.append(f'<path d="M{cx} {cy - r} a{r} {r} 0 0 0 0 {2 * r} z" fill="{col}"/>')
    o.append("</svg>")
    return "".join(o)


# ------------------------------------------------------------------ numbers

def jload(p):
    p = ROOT / p
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def generation_summary() -> dict:
    """Per condition: mean over seeds, then over layouts (coarse granularity), and paired wins against text-only."""
    with open(GEN / "summary.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["granularity"] == "coarse"]
    mets = ("HOR", "HOR_area", "object_recall", "HAR", "foreign_components")
    cell = defaultdict(lambda: defaultdict(list))
    raw = {}
    for r in rows:
        key = (r["layout"], int(r["seed"])) if "layout" in r else (r["example"], int(r["seed"]))
        raw[(r["condition"],) + key] = r
        for m in mets:
            if r[m] not in ("", "None"):
                cell[r["condition"]][(key[0], m)].append(float(r[m]))
    out = {}
    for c, d in cell.items():
        out[c] = {m: statistics.mean(statistics.mean(v) for (ex, mm), v in d.items() if mm == m) for m in mets}
        out[c]["n"] = sum(1 for k in raw if k[0] == c)
    for c in out:
        w = defaultdict(int)
        n = 0
        for (cc, ex, s), r in raw.items():
            b = raw.get(("photo_textonly", ex, s))
            if cc != c or b is None:
                continue
            n += 1
            w["object_recall"] += float(r["object_recall"]) > float(b["object_recall"])
            for m in ("HOR", "HOR_area", "HAR"):
                w[m] += float(r[m]) < float(b[m])
        out[c]["wins"] = dict(w)
        out[c]["pairs"] = n
    return out


def pilot_stats() -> dict:
    """Pilot for H1. The layout (case) is the unit: mean over seeds, then paired differences per case.
    Five cases are descriptive: we report every case and an exact sign-flip p (its minimum with 5 cases is 1/16)."""
    with open(GEN / "summary.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["granularity"] == "coarse"]
    exs = sorted({r["layout"] for r in rows})
    cell = defaultdict(list)
    for r in rows:
        for m in ("object_recall", "HOR", "HAR"):
            cell[(r["condition"], m, r["layout"])].append(float(r[m]))
    M = {(c, m): np.array([np.mean(cell[(c, m, e)]) for e in exs]) for c in {r["condition"] for r in rows} for m in ("object_recall", "HOR", "HAR")}
    signs = np.array(list(itertools.product([1, -1], repeat=len(exs))))
    out = {"n_cases": len(exs), "cases": exs}
    for m in ("object_recall", "HOR", "HAR"):
        t, l, g = M[("photo_textonly", m)], M[("photo_llm", m)], M[("photo", m)]
        d = l - t
        out[m] = {"llm_minus_text": float(d.mean()), "diffs": d.tolist(), "llm_minus_hand": float((l - g).mean()),
                  "p_flip": float((np.abs((signs * d).mean(1)) >= abs(d.mean()) - 1e-12).mean()),
                  "n_better": int((d > 0).sum() if m == "object_recall" else (d < 0).sum())}
    return out


def extraction_summary(pred_name: str) -> dict | None:
    """LLM layouts against the reference layouts, pooled over the examples."""
    rows = []
    for d in sorted(EX.iterdir()):
        p = d / pred_name
        if not p.exists():
            continue
        r = compare(load_layout(p), load_layout(d / "layout.gold.json"), (d / "report.he.txt").read_text(encoding="utf-8"))
        meta = json.loads(p.with_suffix(".meta.json").read_text(encoding="utf-8"))
        r.update(ex=d.name, attempts=len(meta["attempts"]), valid=meta["valid"],
                 model=(meta["attempts"][-1].get("models_used") or ["?"])[0])
        g = load_layout(d / "layout.gold.json")
        r["n_rel_gold"], r["n_rel_pred"] = len(g["relations"]), len(load_layout(p)["relations"])
        rows.append(r)
    if not rows:
        return None
    S = lambda k: sum(r[k] for r in rows)                   # noqa: E731
    tp_r = sum(round(r["relation_recall"] * r["n_rel_gold"]) for r in rows if r["relation_recall"] is not None)
    n_rp = sum(round(tp / r["relation_precision"]) if r["relation_precision"] else 0
               for r in rows for tp in [round((r["relation_recall"] or 0) * r["n_rel_gold"])])
    ious = [r["box_iou"] for r in rows if r["box_iou"] is not None]
    return {"rows": rows, "n": len(rows), "valid": sum(r["valid"] for r in rows), "first_try": sum(r["attempts"] == 1 for r in rows),
            "model": rows[0]["model"], "elements_gold": S("n_gold"), "elements_pred": S("n_pred"), "matched": S("n_matched"),
            "precision": S("n_matched") / S("n_pred"), "recall": S("n_matched") / S("n_gold"),
            "invented": sum(len(r["invented_elements"]) for r in rows), "missed": sum(len(r["missed_elements"]) for r in rows),
            "class_acc": statistics.mean(r["class_accuracy"] for r in rows),
            "rel_recall": tp_r / S("n_rel_gold"), "rel_precision": tp_r / n_rp if n_rp else None,
            "rel_gold": S("n_rel_gold"), "rel_tp": tp_r, "rel_pred": n_rp,
            "box_iou_mean": statistics.mean(ious), "box_iou_min": min(ious), "box_iou_max": max(ious),
            "over_spec": sum((r["over_specification_rate"] or 0) * r["n_term_pairs"] for r in rows),
            "pos_invented": sum(len(r["positionally_invented"]) for r in rows)}


CONDS = [("photo_textonly", "Text only (baseline)", GREY), ("gligen", "GLIGEN, boxes", OCHRE),
         ("photo", "ControlNet, hand-built layout", BLUE), ("photo_llm", "ControlNet, LLM layout", RED)]


def gen_chart(G, PM, w=820, rowh=46, fs=21):
    val = {c: {**G[c], "miou": PM["conditions"][c]["miou"]} for c, _, _ in CONDS}
    mets = [("object_recall", "Object recall: layout elements drawn, higher is better"), ("miou", "mIoU to the text layout (ground, built, pit), higher is better"),
            ("HAR", "HAR: empty area that was filled, lower is better")]
    return "".join(hbar([dict(label=n, v=val[c][m], color=col, bold=c == "photo_llm") for c, n, col in CONDS], w=w, label_w=330, title=t, rowh=rowh, fs=fs) for m, t in mets)


WALL_MODELS = [("all_wall", "Everything is wall", GREY), ("bright_texture", "Bright, textured pixels", GREY), ("segformer_ade", "SegFormer, ADE20K", OCHRE),
               ("grounded_sam", "Grounded-SAM, zero-shot", OCHRE), ("unet_synth", "U-Net, generated only", RED),
               ("unet_real", "U-Net, 14 real tiles", BLUE), ("unet_synth_real", "U-Net, generated + 14 real", RED)]


def wall_chart(W, w=980, rowh=64, fs=22, xmax=.6):
    return hbar([dict(label=n, v=W["models"][k]["iou"], lo=W["models"][k]["iou_ci95"][0], hi=W["models"][k]["iou_ci95"][1], color=c, bold=k.startswith("unet"),
                      tip=f'{n}\nIoU {W["models"][k]["iou"]:.3f}, 95% CI {W["models"][k]["iou_ci95"][0]:.3f} to {W["models"][k]["iou_ci95"][1]:.3f}\nprecision {W["models"][k]["precision"]:.2f}, recall {W["models"][k]["recall"]:.2f}')
                 for k, n, c in WALL_MODELS if k in W["models"]], w=w, label_w=310, xmax=xmax, ticks=[0, .2, .4, .6], rowh=rowh, fs=fs, title="Wall IoU, 95% interval")


def fig_text(ex: str) -> str:
    return (EX / ex / "report.he.txt").read_text(encoding="utf-8").strip()


def he_marks(ex: str) -> str:
    """The Hebrew report with each element's verbatim quote wrapped in <mark data-k>."""
    t = fig_text(ex)
    L = load_layout(EX / ex / "layout.gold.json")
    spans = []
    for e in L["elements"]:
        i = t.find(e["source_span"])
        if i >= 0:
            spans.append((i, i + len(e["source_span"]), e["id"]))
    out, pos = [], 0
    for a, b, k in sorted(spans):
        if a < pos:
            continue
        out += [esc(t[pos:a]), f'<mark data-k="{k}">{esc(t[a:b])}</mark>']
        pos = b
    return "".join(out) + esc(t[pos:])


def zero_tiles() -> dict:
    """Test tiles that contain walls on which a model finds practically none (IoU below 0.005)."""
    with open(ROOT / "04_cv_experiment/annotation/real_walls_manifest.csv", encoding="utf-8") as f:
        has = {r["tile"] for r in csv.DictReader(f) if r["split"] == "test" and float(r["wall_frac"]) > 0}
    with open(ROOT / "results/metrics/real_walls_per_tile.csv", encoding="utf-8") as f:
        out = defaultdict(int)
        for r in csv.DictReader(f):
            out[r["model"]] += r["tile"] in has and float(r["iou"]) < 0.005
    return {**out, "tiles_with_walls": len(has)}


def tile_overlay(tile, model, w=380):
    base = "04_cv_experiment/data/real/walls/test"
    return overlay_mask(f"{base}/images/{tile}.png", f"{base}/masks/{tile}.png", f"results/predictions/real_walls/{model}/{tile}.png", w)


def overlap_counts(ov) -> dict:
    """Overlapping pairs among the 54 tiles that are used (the second copy of each identical photo is removed)."""
    with open(ROOT / "04_cv_experiment/annotation/real_walls_manifest.csv", encoding="utf-8") as f:
        dup = {r["tile"] for r in csv.DictReader(f) if r["split"] == "duplicate"}
    p = [o for o in ov["overlaps"] if o["a"] not in dup and o["b"] not in dup]
    corner = sum(all(v != 0 for v in o["shift_px_of_200"]) for o in p)
    return {"pairs": len(p), "edge": len(p) - corner, "corner": corner, "identical": len(dup)}


def pred_snippet(ex, eid) -> str:
    e = next(x for x in load_layout(EX / ex / "layout.pred.json")["elements"] if x["id"] == eid)
    return json.dumps(e, ensure_ascii=False, indent=1)


def context() -> dict:
    eda = jload("results/eda/eda.json")
    raw = jload("results/eda/eda_raw.json")
    g, w_ = generation_summary(), jload("results/metrics/real_walls.json")
    return dict(GENCHART=lambda **k: gen_chart(g, jload("results/metrics/pilot_miou.json"), **k), WALLCHART=lambda **k: wall_chart(w_, **k), TILE="tile_777_2", img=img, png=png, plan=plan, hbar=hbar, hist1=hist1, hist2=hist2, strips=strips, share_bar=share_bar, heat=heat,
                dotmatrix=dotmatrix, plan_compare=plan_compare, overlay_mask=overlay_mask, label_img=label_img, wall_outline=wall_outline, he_marks=he_marks, esc=esc, np=np, RED=RED, BLUE=BLUE, OCHRE=OCHRE, GREY=GREY, FILL=FILL,
                E=eda, raw=raw, G=g, P=pilot_stats(), ZERO=zero_tiles(), pred_snippet=pred_snippet, RC=lambda c: "/".join(str(jload("results/metrics/error_analysis.json")["generation"][c]["recall_coarse_by_class"][k]["drawn"]) for k in ("wall", "installation", "pit")), ov=tile_overlay, PM=jload("results/metrics/pilot_miou.json"), D=jload("results/metrics/domain_gap_probe.json"), OC=overlap_counts(jload("04_cv_experiment/annotation/real_walls_overlaps.json")), GS=ast.literal_eval(w_["selected_on_dev"]["grounded_sam"]["choice"]), X=extraction_summary("layout.pred.json"), XS=extraction_summary("layout.pred_sonnet.json"),
                S=jload("results/metrics/synth_test.json"), C=jload("results/metrics/synth_test_confusion.json"),
                W=w_, ERR=jload("results/metrics/error_analysis.json"),
                OV=jload("04_cv_experiment/annotation/real_walls_overlaps.json"), pct=lambda v, d=0: f"{100 * v:.{d}f}%", ROOT=ROOT, load_layout=load_layout)


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>{css}</style></head>
<body><div id="stage">{slides}</div><div id="tip"></div>
<div id="ui"><button data-a="prev" aria-label="Previous">&#8592;</button><button data-a="next" aria-label="Next">&#8594;</button><button data-a="over">Overview (O)</button><button data-a="notes">Notes (S)</button><button data-a="full">Full screen (F)</button></div>
<script>{js}</script></body></html>
"""


def build(name: str, ctx: dict) -> None:
    src = (HERE / f"{name}.tpl.html").read_text(encoding="utf-8")
    title = re.search(r"<!--\s*title:\s*(.*?)\s*-->", src).group(1)

    def sub(m):
        try:
            return str(eval(m.group(1).strip(), ctx))       # noqa: S307  our own template, not user input
        except Exception as exc:
            raise SystemExit(f"{name}: {{{{ {m.group(1).strip()[:120]} }}}} -> {type(exc).__name__}: {exc}")

    body = re.sub(r"\{\{(.+?)\}\}", sub, re.sub(r"<!--.*?-->", "", src, flags=re.S), flags=re.S)
    body = re.sub(r"(<section class=\"slide.*?</section>)", r'<div class="cell">\1</div>', body, flags=re.S)
    out = PAGE.format(title=esc(title), css=(HERE / "deck.css").read_text(encoding="utf-8"),
                      js=(HERE / "deck.js").read_text(encoding="utf-8"), slides=body)
    dst = (HERE / "build" if name == "visuals" else HERE.parent) / f"{name}.html"
    dst.parent.mkdir(exist_ok=True)
    dst.write_text(out, encoding="utf-8")
    print(f"{dst.name}: {body.count('<section ')} slides, {dst.stat().st_size / 1e6:.1f} MB")


def build_readme(ctx: dict) -> None:
    src = (HERE / "README.tpl.md").read_text(encoding="utf-8")
    (ROOT / "README.md").write_text(re.sub(r"\{\{(.+?)\}\}", lambda m: str(eval(m.group(1).strip(), ctx)), src, flags=re.S), encoding="utf-8")  # noqa: S307
    print("README.md written")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    c = context()
    for n in [a for a in sys.argv[1:] if a != "readme"] or ([] if sys.argv[1:] else ["proposal", "midterm", "visuals"]):
        build(n, c)
    if not sys.argv[1:] or "readme" in sys.argv[1:]:
        build_readme(c)
