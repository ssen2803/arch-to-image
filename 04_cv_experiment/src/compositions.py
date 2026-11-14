"""תיוג קומפוזיציות מרחביות מתוך מסכת תוויות, לניסוי ההכללה (מקרה קצה 4).

התיוג נעשה מהמסכה ולא מה-Layout, כדי שאותה פונקציה תעבוד גם על הדאטהסט הסינתטי
וגם על קבוצת המבחן האמיתית המתויגת.

תגיות טופולוגיות (לא תלויות בכיוון התמונה):
  pit_in_floor / pit_in_open                   - מה מקיף את הבור (רוב הטבעת סביבו)
  installation_on_floor / installation_in_open - מה מקיף את המתקן
תגיות כיווניות (במסגרת התמונה; צפון = למעלה רק בתמונה סינתטית או בצילום מיושר לצפון):
  wall_south_of_pit / wall_north_of_pit        - קיר מתחת / מעל לבור, עם חפיפה אופקית וקרוב אליו

אזהרה מתודולוגית: אוגמנטציה של סיבובים והיפוכים הופכת "קיר מדרום לבור" ל"קיר מצפון לבור".
לכן בניסוי על תגית כיוונית חובה לאמן עם --no-geometric-aug, אחרת הקומפוזיציה ה"חדשה" נראתה באימון.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

BG, FLOOR, WALL, INST, PIT = 0, 1, 2, 3, 4
TOPOLOGICAL = ["pit_in_floor", "pit_in_open", "installation_on_floor", "installation_in_open"]
DIRECTIONAL = ["wall_south_of_pit", "wall_north_of_pit"]
ALL_TAGS = TOPOLOGICAL + DIRECTIONAL

_EIGHT = np.ones((3, 3), bool)


def _components(mask: np.ndarray, min_area: int):
    lab, _ = ndimage.label(mask, structure=_EIGHT)
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        m = lab[sl] == i
        if m.sum() >= min_area:
            full = np.zeros(mask.shape, bool)
            full[sl] = m
            yield full, sl


def _surround(label: np.ndarray, comp: np.ndarray, ring_px: int) -> np.ndarray:
    ring = ndimage.binary_dilation(comp, structure=_EIGHT, iterations=ring_px) & ~comp
    vals = label[ring]
    return vals[vals != 255]


def composition_tags(label: np.ndarray, min_area: int = 64, ring_px: int = 6, near_px: int = 64) -> set[str]:
    tags: set[str] = set()
    walls = list(_components(label == WALL, min_area))
    for cls, name in ((PIT, "pit"), (INST, "installation")):
        for comp, sl in _components(label == cls, min_area):
            around = _surround(label, comp, ring_px)
            if around.size:
                share_floor = (around == FLOOR).mean()
                share_bg = (around == BG).mean()
                if share_floor >= 0.5:
                    tags.add(f"{name}_in_floor" if cls == PIT else f"{name}_on_floor")
                elif share_bg >= 0.5:
                    tags.add(f"{name}_in_open")
            if cls != PIT:
                continue
            py0, py1, px0, px1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
            for wmask, wsl in walls:
                wy0, wy1, wx0, wx1 = wsl[0].start, wsl[0].stop, wsl[1].start, wsl[1].stop
                if min(px1, wx1) - max(px0, wx0) <= 0:  # אין חפיפה אופקית
                    continue
                if 0 <= wy0 - py1 <= near_px:           # הקיר מתחת לבור בתמונה = מדרום לו
                    tags.add("wall_south_of_pit")
                if 0 <= py0 - wy1 <= near_px:           # הקיר מעל לבור = מצפון לו
                    tags.add("wall_north_of_pit")
    return tags
