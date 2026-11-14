"""מחולל Layouts סינתטיים לפי האונטולוגיה של שלב 2.

כל Layout שנוצר עובר את אותו מאמת של שלב 2 (סכמה + גיאומטריה), ולכן נתוני האימון
כפופים לאותם כללים כמו ה-Layouts שמגיעים מטקסט. שתי תבניות:
  rooms - מבנה של 1-3 חדרים: קירות היקפיים, מחיצה, פתחים, רצפות, מתקנים ובורות.
  open  - שטח פתוח: כתמי רצפה, בורות, מתקנים, ולפעמים קטע קיר שחוצה את השטח.
מבנים נמשכים לעתים אל מעבר לגבול החפירה (truncated), כמו בדו"חות אמיתיים.
"""

from __future__ import annotations

import random

from common import NUM_CLASSES  # noqa: F401  (מוודא שנתיבי שלבים 2-3 נטענו)
from layout_core import Box, errors_only, validate

PERIODS = ["Iron Age II", "Bronze Age", "Hellenistic", "Roman", "Byzantine", "Early Islamic", "Chalcolithic"]
FLOORS = [("beaten-earth floor", "beaten earth floor"), ("white plaster floor", "white plaster floor"),
          ("flagstone pavement", "flagstone pavement"), ("coarse mosaic floor", "coarse white mosaic floor")]
ROUND_INST = [("tabun", "clay tabun oven"), ("hearth", "stone-lined hearth"), ("round basin", "round stone basin")]
PITS = [("pit", "round pit"), ("refuse pit", "refuse pit"), ("silo", "round storage silo pit")]


class _Builder:
    def __init__(self, size: float, rng: random.Random):
        self.S, self.rng = size, rng
        self.elements: list[dict] = []
        self.count = {"wall": 0, "floor": 0, "installation": 0, "pit": 0}

    def add(self, cls: str, x0: float, y0: float, x1: float, y1: float, shape: str = "rect",
            subtype: str = "", phrase: str = "", min_size: float = 0.3) -> dict | None:
        """מוסיף אלמנט אחרי חיתוך לגבולות השטח. צלע שנחתכה מסומנת כ-truncated."""
        S = self.S
        trunc = [s for s, cut in (("W", x0 < 0), ("S", y0 < 0), ("E", x1 > S), ("N", y1 > S)) if cut]
        x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(S, x1), min(S, y1)
        if x1 - x0 < min_size or y1 - y0 < min_size:
            return None
        self.count[cls] += 1
        prefix = {"wall": "W", "floor": "F", "installation": "I", "pit": "P"}[cls]
        el = {"id": f"{prefix}{self.count[cls]}", "class": cls, "subtype": subtype or cls, "phrase": phrase or cls,
              "locus": None, "evidence": "attested",
              "bbox": {"x": round(x0, 3), "y": round(y0, 3), "w": round(x1 - x0, 3), "h": round(y1 - y0, 3)},
              "shape": shape, "dims_source": "stated", "position": "stated", "truncated": trunc,
              "source_span": "synthetic"}
        self.elements.append(el)
        return el

    def overlaps(self, x0, y0, x1, y1, classes=("wall", "installation")) -> bool:
        b = Box(x0, y0, x1, y1)
        return any(e["class"] in classes and b.intersection_area(Box.from_bbox(e["bbox"])) > 0
                   for e in self.elements)


def _wall_with_door(b: _Builder, horizontal: bool, fixed: float, a0: float, a1: float, t: float, door_p: float):
    """קיר לאורך ציר אחד (a0..a1), עם פתח אופציונלי שמפצל אותו לשני קטעים."""
    rng = b.rng
    segments = [(a0, a1)]
    if rng.random() < door_p and a1 - a0 > 3.0:
        d = rng.uniform(0.8, 1.2)
        c = rng.uniform(a0 + 1.0, a1 - 1.0 - d)
        segments = [(a0, c), (c + d, a1)]
    for s0, s1 in segments:
        if horizontal:
            b.add("wall", s0, fixed, s1, fixed + t, subtype="stone wall", phrase="stone wall")
        else:
            b.add("wall", fixed, s0, fixed + t, s1, subtype="stone wall", phrase="stone wall")


def _rooms(b: _Builder):
    rng, S = b.rng, b.S
    t = rng.uniform(0.5, 0.9)
    # טביעת הרגל של המבנה, לעתים חורגת מגבול החפירה
    x0 = rng.uniform(-2.0, S * 0.25)
    y0 = rng.uniform(-2.0, S * 0.25)
    x1 = rng.uniform(S * 0.65, S + 2.0)
    y1 = rng.uniform(S * 0.65, S + 2.0)
    for horizontal, fixed, a0, a1 in ((True, y0, x0, x1), (True, y1 - t, x0, x1),
                                      (False, x0, y0 + t, y1 - t), (False, x1 - t, y0 + t, y1 - t)):
        if -t < fixed < S:  # צלע שנמצאת בתוך השטח (אחרת היא מחוץ לחפירה)
            _wall_with_door(b, horizontal, fixed, a0, a1, t, door_p=0.35)
    rooms = [(x0 + t, y0 + t, x1 - t, y1 - t)]
    if rng.random() < 0.6:  # קיר מחיצה
        rx0, ry0, rx1, ry1 = rooms[0]
        if rng.random() < 0.5 and rx1 - rx0 > 4:
            px = rng.uniform(rx0 + 0.35 * (rx1 - rx0), rx0 + 0.65 * (rx1 - rx0))
            _wall_with_door(b, False, px, ry0, ry1, t * 0.85, door_p=0.7)
            rooms = [(rx0, ry0, px, ry1), (px + t * 0.85, ry0, rx1, ry1)]
        elif ry1 - ry0 > 4:
            py = rng.uniform(ry0 + 0.35 * (ry1 - ry0), ry0 + 0.65 * (ry1 - ry0))
            _wall_with_door(b, True, py, rx0, rx1, t * 0.85, door_p=0.7)
            rooms = [(rx0, ry0, rx1, py), (rx0, py + t * 0.85, rx1, ry1)]
    for rx0, ry0, rx1, ry1 in rooms:
        if rng.random() < 0.85:
            sub, phr = rng.choice(FLOORS)
            b.add("floor", rx0, ry0, rx1, ry1, subtype=sub, phrase=phr)
        _furnish(b, max(rx0, 0), max(ry0, 0), min(rx1, S), min(ry1, S), n_inst=rng.randint(0, 2),
                 n_pit=rng.choice([0, 0, 1, 2]))
    # בורות מחוץ למבנה
    _pits(b, 0, 0, S, S, rng.choice([0, 1, 2]), avoid_floors=True)


def _open(b: _Builder):
    rng, S = b.rng, b.S
    for _ in range(rng.randint(0, 2)):
        w, h = rng.uniform(2, S * 0.7), rng.uniform(2, S * 0.7)
        x, y = rng.uniform(-1, S - w + 1), rng.uniform(-1, S - h + 1)
        sub, phr = rng.choice(FLOORS)
        b.add("floor", x, y, x + w, y + h, subtype=sub, phrase=phr)
    if rng.random() < 0.4:  # קטע קיר שחוצה את השטח
        t = rng.uniform(0.5, 0.9)
        if rng.random() < 0.5:
            y = rng.uniform(0.5, S - 1.5)
            b.add("wall", rng.uniform(-2, S * 0.3), y, rng.uniform(S * 0.6, S + 2), y + t,
                  subtype="stone wall", phrase="stone wall")
        else:
            x = rng.uniform(0.5, S - 1.5)
            b.add("wall", x, rng.uniform(-2, S * 0.3), x + t, rng.uniform(S * 0.6, S + 2),
                  subtype="stone wall", phrase="stone wall")
    _furnish(b, 0, 0, S, S, n_inst=rng.randint(0, 2), n_pit=0)
    _pits(b, 0, 0, S, S, rng.randint(1, 5), avoid_floors=False)


def _furnish(b: _Builder, x0, y0, x1, y1, n_inst: int, n_pit: int):
    rng = b.rng
    for _ in range(n_inst):
        for _attempt in range(20):
            if rng.random() < 0.6:  # עגול
                d = rng.uniform(0.6, 1.1)
                if x1 - x0 < d + 0.4 or y1 - y0 < d + 0.4:
                    break
                x, y = rng.uniform(x0 + 0.2, x1 - d - 0.2), rng.uniform(y0 + 0.2, y1 - d - 0.2)
                if not b.overlaps(x, y, x + d, y + d, ("wall", "installation", "pit")):
                    sub, phr = rng.choice(ROUND_INST)
                    b.add("installation", x, y, x + d, y + d, "ellipse", sub, phr)
                    break
            else:  # ספסל או אגן מלבני
                w, h = (rng.uniform(1.2, 2.5), rng.uniform(0.4, 0.6))
                if rng.random() < 0.5:
                    w, h = h, w
                if x1 - x0 < w + 0.2 or y1 - y0 < h + 0.2:
                    continue
                x, y = rng.uniform(x0 + 0.05, x1 - w - 0.05), rng.uniform(y0 + 0.05, y1 - h - 0.05)
                if not b.overlaps(x, y, x + w, y + h, ("wall", "installation", "pit")):
                    b.add("installation", x, y, x + w, y + h, "rect", "stone bench", "low stone bench")
                    break
    _pits(b, x0, y0, x1, y1, n_pit, avoid_floors=False)


def _pits(b: _Builder, x0, y0, x1, y1, n: int, avoid_floors: bool):
    rng = b.rng
    avoid = ("wall", "installation", "pit", "floor") if avoid_floors else ("wall", "installation", "pit")
    for _ in range(n):
        for _attempt in range(20):
            d = rng.uniform(0.6, 1.8)
            if x1 - x0 < d or y1 - y0 < d:
                break
            x, y = rng.uniform(x0, x1 - d), rng.uniform(y0, y1 - d)
            if not b.overlaps(x, y, x + d, y + d, avoid):
                sub, phr = rng.choice(PITS)
                b.add("pit", x, y, x + d, y + d, "ellipse", sub, phr)
                break


def generate_layout(seed: int, max_tries: int = 50) -> dict:
    """Layout סינתטי תקין (ללא שגיאות אימות). דטרמיניסטי לפי seed."""
    rng = random.Random(seed)
    for attempt in range(max_tries):
        S = rng.choice([5, 6, 7, 8, 10])
        b = _Builder(S, rng)
        template = "rooms" if rng.random() < 0.6 else "open"
        (_rooms if template == "rooms" else _open)(b)
        if not b.elements:
            continue
        layout = {
            "schema_version": "1.1",
            "source": {"id": f"synth_{seed:05d}", "language": "en", "period": rng.choice(PERIODS),
                       "context": "domestic building" if template == "rooms" else "open-air activity area"},
            "area": {"label": "synthetic", "width_m": S, "height_m": S, "dims_source": "stated"},
            "elements": b.elements,
            "relations": [],
            "unmapped_mentions": [],
        }
        if not errors_only(validate(layout)):
            return layout
    raise RuntimeError(f"לא נוצר Layout תקין עבור seed={seed} אחרי {max_tries} ניסיונות")
