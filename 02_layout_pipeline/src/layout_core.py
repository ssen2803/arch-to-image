"""ליבת ה-Layout: סכמה, גיאומטריה, ואימות סמנטי של קובצי ArchLayout (גרסאות 1.1 ו-1.2).

האימות מתבצע בשתי שכבות:
1. מבני - מול JSON Schema (טיפוסים, שדות חובה, enum).
2. סמנטי/גיאומטרי - האם ה-Layout "תקין" במובן המרחבי: מזהים ייחודיים,
   תיבות בתוך שטח החפירה, צלעות קטועות שנוגעות בגבול, וכל יחס מרחבי
   שהוצהר אכן מתקיים בגיאומטריה.

מערכת הצירים: מטרים, ראשית בפינה הדרום-מערבית, x מזרחה, y צפונה.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "schema" / "arch_layout.schema.json"
THESAURUS_PATH = ROOT / "schema" / "thesaurus.json"

# שדות שנוספו בגרסה 1.2. מסמך שמצהיר 1.1 אינו רשאי להכיל אותם.
V12_FIELDS = {"source": {"site", "report"}, "area": {"georef"},
              "element": {"term", "attributes", "dating", "elevation", "phase"}}

# סובלנות גיאומטרית במטרים - מתאימה לדיוק מקובל בתיאורי חפירה (עשרות ס"מ).
DEFAULT_TOL_M = 0.3

# האונטולוגיה: 5 מחלקות. background אינו אלמנט - הוא כל מה שלא מכוסה.
CLASSES = ["background", "floor", "wall", "installation", "pit"]
CLASS_PREFIX = {"wall": "W", "floor": "F", "installation": "I", "pit": "P"}

# סדר ציור (z-order): רצפה בתחתית, בורות למעלה - כי בור חותך רצפה/קיר.
Z_ORDER = {"floor": 1, "wall": 2, "installation": 3, "pit": 4}

# פלטת צבעים קבועה - משמשת גם לאיור הסכמטי וגם למסכת הסגמנטציה (GT ל-mIoU).
PALETTE = {
    "background": (240, 234, 222),
    "floor": (214, 178, 112),
    "wall": (92, 64, 51),
    "installation": (196, 78, 56),
    "pit": (58, 90, 120),
}

DIRECTIONAL = {"north_of", "south_of", "east_of", "west_of"}
OPPOSITE = {"north_of": "south_of", "south_of": "north_of", "east_of": "west_of", "west_of": "east_of"}


@dataclass(frozen=True)
class Issue:
    level: str  # "error" | "warning"
    where: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.where}: {self.message}"


@dataclass(frozen=True)
class Box:
    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def from_bbox(cls, b: dict) -> "Box":
        return cls(b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"])

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    def intersection_area(self, o: "Box") -> float:
        dx = min(self.x1, o.x1) - max(self.x0, o.x0)
        dy = min(self.y1, o.y1) - max(self.y0, o.y0)
        return dx * dy if dx > 0 and dy > 0 else 0.0

    def gap(self, o: "Box") -> float:
        """מרחק אוקלידי בין שתי תיבות (0 אם הן נוגעות או חופפות)."""
        dx = max(0.0, o.x0 - self.x1, self.x0 - o.x1)
        dy = max(0.0, o.y0 - self.y1, self.y0 - o.y1)
        return math.hypot(dx, dy)

    def within(self, o: "Box", tol: float) -> bool:
        return (self.x0 >= o.x0 - tol and self.y0 >= o.y0 - tol
                and self.x1 <= o.x1 + tol and self.y1 <= o.y1 + tol)


# ---------------------------------------------------------------- סכמה

def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def load_layout(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


_THESAURUS: dict | None = None


def load_thesaurus() -> dict:
    global _THESAURUS
    if _THESAURUS is None:
        _THESAURUS = json.loads(THESAURUS_PATH.read_text(encoding="utf-8"))
    return _THESAURUS


def term_ancestors(term: str) -> list[str]:
    """המונח ואבותיו עד שורש המחלקה, למשל posthole -> socket -> pit."""
    terms = load_thesaurus()["terms"]
    chain = []
    while term is not None and term not in chain:
        chain.append(term)
        term = terms[term]["broader"] if term in terms else None
    return chain


def is_class_root(term: str) -> bool:
    return load_thesaurus()["terms"].get(term, {}).get("broader", "x") is None


def _surface_forms(term_or_material: str, kind: str, language: str) -> list[str]:
    entry = load_thesaurus()[kind].get(term_or_material, {})
    return entry.get("he", []) if language == "he" else [term_or_material]


def _appears_in(forms: list[str], span: str) -> bool:
    span = span.lower()
    return any(f.lower() in span for f in forms)


_UNSUPPORTED_BY_API = {
    "$schema", "$id", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "multipleOf", "minLength", "maxLength", "pattern", "minItems", "maxItems", "uniqueItems",
}


def api_schema(schema: dict | None = None) -> dict:
    """גרסה של הסכמה שמתאימה ל-structured outputs של ה-API.

    מסירה אילוצים מספריים/מחרוזתיים שה-API עשוי לא לתמוך בהם, וממירה
    type-רשימה ל-anyOf ו-const ל-enum. האילוצים שהוסרו נאכפים מקומית
    ע"י validate() - לכן אין אובדן קפדנות, רק העברה של האכיפה לצד הלקוח.
    """
    schema = copy.deepcopy(schema or load_schema())

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                if k in _UNSUPPORTED_BY_API:
                    continue
                out[k] = walk(v)
            if isinstance(out.get("type"), list):
                types = out.pop("type")
                out["anyOf"] = [{"type": t} for t in types]
            if "const" in out:
                out["enum"] = [out.pop("const")]
            return out
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    out = walk(schema)
    # מונחים: הרשימה הסגורה נמצאת ב-thesaurus.json. ב-API היא נאכפת כ-enum, ומקומית ב-validate_semantics.
    term_value = out.get("$defs", {}).get("term", {}).get("properties", {}).get("value")
    if term_value is not None:
        term_value["enum"] = sorted(load_thesaurus()["terms"])
        term_value.pop("type", None)
    return out


# ---------------------------------------------------------------- אימות

def validate_structure(layout: dict, schema: dict | None = None) -> list[Issue]:
    validator = Draft202012Validator(schema or load_schema())
    issues = []
    for err in sorted(validator.iter_errors(layout), key=lambda e: list(e.absolute_path)):
        where = "/".join(str(p) for p in err.absolute_path) or "(root)"
        issues.append(Issue("error", where, f"הפרת סכמה: {err.message}"))
    return issues


def check_relation(pred: str, a: Box, b: Box, tol: float) -> tuple[bool, str]:
    """בודק אם יחס מתקיים בגיאומטריה. מחזיר (תקין?, הסבר)."""
    if pred == "north_of":
        return a.cy > b.cy, f"מרכז y של הנושא {a.cy:.2f} לעומת {b.cy:.2f}"
    if pred == "south_of":
        return a.cy < b.cy, f"מרכז y של הנושא {a.cy:.2f} לעומת {b.cy:.2f}"
    if pred == "east_of":
        return a.cx > b.cx, f"מרכז x של הנושא {a.cx:.2f} לעומת {b.cx:.2f}"
    if pred == "west_of":
        return a.cx < b.cx, f"מרכז x של הנושא {a.cx:.2f} לעומת {b.cx:.2f}"
    if pred == "inside":
        return a.within(b, tol), "תיבת הנושא אינה מוכלת בתיבת המושא"
    if pred == "on":
        ratio = a.intersection_area(b) / a.area if a.area else 0.0
        return ratio >= 0.5, f"רק {ratio:.0%} משטח הנושא נמצא על המושא (נדרש 50% לפחות)"
    if pred == "adjacent_to":
        g = a.gap(b)
        return g <= tol, f"מרווח של {g:.2f} מ' בין התיבות (סובלנות {tol} מ')"
    if pred == "cuts":
        return a.intersection_area(b) > 0, "התיבות אינן חופפות, ולכן הנושא אינו יכול לחתוך את המושא"
    if pred == "under":
        return a.intersection_area(b) > 0, "התיבות אינן חופפות, ולכן המושא אינו יכול לכסות את הנושא"
    return False, f"פרדיקט לא מוכר: {pred}"


# יחסים בעלי משמעות סטרטיגרפית: (נושא, מושא) -> מי מעל מי
def _above_edges(layout: dict) -> list[tuple[str, str]]:
    """רשימת זוגות (עליון, תחתון) מתוך cuts / on / under."""
    edges = []
    for r in layout["relations"]:
        s, p, o = r["subject"], r["predicate"], r["object"]
        if p in ("cuts", "on"):
            edges.append((s, o))
        elif p == "under":
            edges.append((o, s))
    return edges


def stratigraphic_order(layout: dict) -> list[dict] | None:
    """סדר ציור מלמטה למעלה. מכבד את היחסים הסטרטיגרפיים, ובין אלמנטים שאין ביניהם
    יחס - לפי סדר המחלקות (רצפה, קיר, מתקן, בור). מחזיר None אם יש מעגל."""
    import heapq
    els = {e["id"]: e for e in layout["elements"]}
    below_count = {i: 0 for i in els}
    above_of = {i: [] for i in els}
    for top, bottom in _above_edges(layout):
        if top in els and bottom in els:
            above_of[bottom].append(top)
            below_count[top] += 1
    heap = [(Z_ORDER[els[i]["class"]], i) for i, c in below_count.items() if c == 0]
    heapq.heapify(heap)
    order = []
    while heap:
        _, i = heapq.heappop(heap)
        order.append(els[i])
        for j in above_of[i]:
            below_count[j] -= 1
            if below_count[j] == 0:
                heapq.heappush(heap, (Z_ORDER[els[j]["class"]], j))
    return order if len(order) == len(els) else None


def uncertainty_box(layout: dict, el: dict) -> Box:
    """האזור שבו יכול להימצא אלמנט שמיקומו לא נאמר: חיתוך כל האילוצים שכן נאמרו.

    בלי אילוצים: כל שטח החפירה. inside/on/cuts/under -> תיבת המושא. כיווני -> חצי-מישור.
    adjacent_to -> תיבת המושא מורחבת בגודל האלמנט.
    """
    W, H = layout["area"]["width_m"], layout["area"]["height_m"]
    x0, y0, x1, y1 = 0.0, 0.0, float(W), float(H)
    boxes = {e["id"]: Box.from_bbox(e["bbox"]) for e in layout["elements"]}
    size = max(el["bbox"]["w"], el["bbox"]["h"])
    for r in layout["relations"]:
        if r["subject"] == el["id"] and r["object"] in boxes:
            b, p = boxes[r["object"]], r["predicate"]
        elif r["object"] == el["id"] and r["subject"] in boxes and r["predicate"] in OPPOSITE:
            b, p = boxes[r["subject"]], OPPOSITE[r["predicate"]]  # X north_of el  =>  el south_of X
        else:
            continue
        if p in ("inside", "on", "cuts", "under"):
            x0, y0, x1, y1 = max(x0, b.x0), max(y0, b.y0), min(x1, b.x1), min(y1, b.y1)
        elif p == "north_of":
            y0 = max(y0, b.cy)
        elif p == "south_of":
            y1 = min(y1, b.cy)
        elif p == "east_of":
            x0 = max(x0, b.cx)
        elif p == "west_of":
            x1 = min(x1, b.cx)
        elif p == "adjacent_to":
            x0, y0 = max(x0, b.x0 - size), max(y0, b.y0 - size)
            x1, y1 = min(x1, b.x1 + size), min(y1, b.y1 + size)
    if x1 <= x0 or y1 <= y0:  # אילוצים סותרים - נופלים חזרה לכל השטח (המאמת מתריע על כך בנפרד)
        return Box(0.0, 0.0, float(W), float(H))
    return Box(x0, y0, x1, y1)


def validate_semantics(layout: dict, tol: float = DEFAULT_TOL_M) -> list[Issue]:
    issues: list[Issue] = []
    area = Box(0.0, 0.0, layout["area"]["width_m"], layout["area"]["height_m"])
    elements = layout["elements"]

    # מזהים ייחודיים וקידומת תואמת למחלקה
    boxes: dict[str, Box] = {}
    classes: dict[str, str] = {}
    for i, el in enumerate(elements):
        where = f"elements/{i} ({el['id']})"
        if el["id"] in boxes:
            issues.append(Issue("error", where, "מזהה כפול"))
            continue
        if not el["id"].startswith(CLASS_PREFIX[el["class"]]):
            issues.append(Issue("error", where,
                                f"הקידומת אינה תואמת למחלקה '{el['class']}' (נדרש {CLASS_PREFIX[el['class']]})"))
        box = Box.from_bbox(el["bbox"])
        boxes[el["id"]] = box
        classes[el["id"]] = el["class"]

        if not box.within(area, tol):
            issues.append(Issue("error", where,
                                f"התיבה ({box.x0:.2f},{box.y0:.2f})-({box.x1:.2f},{box.y1:.2f}) "
                                f"חורגת משטח החפירה {area.x1}x{area.y1} מ'"))

        # צלע קטועה חייבת לגעת בגבול החפירה באותו צד
        edge_dist = {"N": abs(area.y1 - box.y1), "S": abs(box.y0 - area.y0),
                     "E": abs(area.x1 - box.x1), "W": abs(box.x0 - area.x0)}
        for side in el["truncated"]:
            if edge_dist[side] > tol:
                issues.append(Issue("error", where,
                                    f"מסומן כנמשך מעבר לגבול {side} אך רחוק ממנו {edge_dist[side]:.2f} מ'"))

        if el["class"] == "wall":
            w, h = el["bbox"]["w"], el["bbox"]["h"]
            if max(w, h) / min(w, h) < 2:
                issues.append(Issue("warning", where, "קיר שאינו מוארך (יחס אורך/רוחב < 2) - לבדוק מידות"))
        if el["evidence"] == "hypothetical" and el["dims_source"] == "stated":
            issues.append(Issue("warning", where, "אלמנט היפותטי עם מידות 'stated' - חריג, לבדוק"))

    # יחסים: קיום מזהים, אין יחס עצמי, התקיימות גיאומטרית, אין סתירות
    seen: set[tuple[str, str, str]] = set()
    for j, rel in enumerate(layout["relations"]):
        s, p, o = rel["subject"], rel["predicate"], rel["object"]
        where = f"relations/{j} ({s} {p} {o})"
        missing = [x for x in (s, o) if x not in boxes]
        if missing:
            issues.append(Issue("error", where, f"מזהה לא קיים: {', '.join(missing)}"))
            continue
        if s == o:
            issues.append(Issue("error", where, "יחס של אלמנט לעצמו"))
            continue
        ok, why = check_relation(p, boxes[s], boxes[o], tol)
        if not ok:
            issues.append(Issue("error", where, f"היחס אינו מתקיים בגיאומטריה: {why}"))
        if p == "cuts" and classes[s] != "pit":
            issues.append(Issue("warning", where, "בדרך כלל רק בור (pit) חותך אלמנט אחר"))
        if p in DIRECTIONAL and ((o, p, s) in seen or (s, OPPOSITE[p], o) in seen):
            issues.append(Issue("error", where, "סתירה מול יחס כיווני קודם"))
        seen.add((s, p, o))

    # סדר סטרטיגרפי: cuts / on / under חייבים ליצור גרף ללא מעגלים
    if not any(i.level == "error" and "מזהה לא קיים" in i.message for i in issues):
        if stratigraphic_order(layout) is None:
            issues.append(Issue("error", "relations", "מעגל סטרטיגרפי: אלמנט נמצא גם מעל וגם מתחת לאחר "
                                                     "(דרך cuts / on / under)"))

    # אלמנט בלי מיקום: לוודא שמציין המקום שלו נמצא באזור האפשרי
    for i, el in enumerate(elements):
        if el["position"] == "unspecified":
            region = uncertainty_box(layout, el)
            if not Box.from_bbox(el["bbox"]).within(region, tol):
                issues.append(Issue("warning", f"elements/{i} ({el['id']})",
                                    "מציין המקום של אלמנט בלי מיקום נמצא מחוץ לאזור שהיחסים מאפשרים"))

    issues += validate_v12(layout)
    return issues


def validate_v12(layout: dict) -> list[Issue]:
    """בדיקות של שדות 1.2: גרסה, מונחים מאוצר המונחים, ספציפיות לפי הראיה, תיארוך וגבהים.

    עיקרון הענווה: מונח ספציפי מסומן stated רק אם הוא מופיע בציטוט. אחרת זו "ספציפיות-יתר",
    כלומר המחלץ (או המתייג) אמר יותר ממה שהטקסט אומר, וזו הזיה עדינה.
    """
    issues: list[Issue] = []
    version = layout.get("schema_version")
    if version == "1.1":
        found = sorted({f for f in V12_FIELDS["source"] if f in layout["source"]}
                       | {f for f in V12_FIELDS["area"] if f in layout["area"]}
                       | {f for el in layout["elements"] for f in V12_FIELDS["element"] if f in el})
        if found:
            issues.append(Issue("error", "schema_version", f"שדות של גרסה 1.2 במסמך 1.1: {', '.join(found)}"))
        return issues

    terms = load_thesaurus()["terms"]
    lang = layout["source"]["language"]
    for i, el in enumerate(layout["elements"]):
        where = f"elements/{i} ({el['id']})"
        term = el.get("term")
        if not term:
            issues.append(Issue("warning", where, "אלמנט בגרסה 1.2 בלי term (אפשר את שורש המחלקה)"))
        if term:
            value = term["value"]
            if value not in terms:
                issues.append(Issue("error", where, f"המונח '{value}' אינו באוצר המונחים"))
            elif terms[value]["class"] != el["class"]:
                issues.append(Issue("error", where,
                                    f"המונח '{value}' שייך למחלקה '{terms[value]['class']}' ולא '{el['class']}'"))
            elif term["basis"] == "stated" and not is_class_root(value) \
                    and not _appears_in(_surface_forms(value, "terms", lang), el["source_span"]):
                issues.append(Issue("warning", where,
                                    f"ספציפיות-יתר: המונח '{value}' מסומן stated אך אינו מופיע בציטוט"))
            # מונח ספציפי עם basis=inferred הוא הצהרה כנה ולא שגיאה. הוא נמדד ב-compare_layouts מול ה-gold.
        material = el.get("attributes", {}).get("material")
        if material and material["basis"] == "stated" \
                and not _appears_in(_surface_forms(material["value"], "materials", lang), el["source_span"]):
            issues.append(Issue("warning", where,
                                f"החומר '{material['value']}' מסומן stated אך אינו מופיע בציטוט"))
        dating = el.get("dating")
        if dating and dating.get("start_year") is not None and dating.get("end_year") is not None \
                and dating["start_year"] > dating["end_year"]:
            issues.append(Issue("error", where, "תיארוך: שנת ההתחלה מאוחרת משנת הסיום"))
        elev = el.get("elevation")
        if elev and elev.get("top_m") is not None and elev.get("bottom_m") is not None \
                and elev["top_m"] < elev["bottom_m"]:
            issues.append(Issue("error", where, "גובה: הראש נמוך מהבסיס"))

    # יחס "מתחת ל-" מול גבהים: אלמנט שמתחת לאחר לא יכול שראשו יהיה גבוה מראש האלמנט שמעליו
    by_id = {el["id"]: el for el in layout["elements"]}
    for j, rel in enumerate(layout["relations"]):
        if rel["predicate"] != "under" or rel["subject"] not in by_id or rel["object"] not in by_id:
            continue
        lo, hi = by_id[rel["subject"]].get("elevation"), by_id[rel["object"]].get("elevation")
        if lo and hi and lo["datum"] == hi["datum"] and lo.get("top_m") is not None \
                and hi.get("top_m") is not None and lo["top_m"] > hi["top_m"] + 0.05:
            issues.append(Issue("warning", f"relations/{j}",
                                f"{rel['subject']} מתחת ל-{rel['object']}, אך ראשו גבוה יותר "
                                f"({lo['top_m']} מול {hi['top_m']} מ')"))
    return issues


def validate(layout: dict, tol: float = DEFAULT_TOL_M) -> list[Issue]:
    """אימות מלא. אם יש הפרות סכמה - לא ממשיכים לבדיקה הסמנטית (המבנה לא אמין)."""
    structural = validate_structure(layout)
    if structural:
        return structural
    return validate_semantics(layout, tol)


def errors_only(issues: list[Issue]) -> list[Issue]:
    return [i for i in issues if i.level == "error"]


def _norm(s: str) -> str:
    return " ".join(s.split())


def check_spans(layout: dict, report_text: str) -> list[Issue]:
    """עקיבות (paradata): כל ציטוט חייב להופיע מילה במילה בטקסט המקור.

    זו ההגנה העיקרית מפני "הזיות" של ה-LLM - אלמנט שאין לו עיגון
    בטקסט לא אמור להיכנס ל-Layout (למעט רווחים, שמנורמלים).
    """
    text = _norm(report_text)
    issues = []
    spans = [(f"elements/{i} ({e['id']})", e["source_span"]) for i, e in enumerate(layout["elements"])]
    spans += [(f"relations/{j}", r["source_span"]) for j, r in enumerate(layout["relations"]) if r["source_span"]]
    spans += [(f"unmapped_mentions/{k}", u["text"]) for k, u in enumerate(layout["unmapped_mentions"])]
    for where, span in spans:
        if _norm(span) not in text:
            issues.append(Issue("error", where, f"הציטוט אינו מופיע בטקסט המקור: \"{span[:60]}\""))
    return issues


def find_report_text(layout_path: Path) -> Path | None:
    """מחפש קובץ report.*.txt באותה תיקייה של ה-Layout."""
    candidates = sorted(Path(layout_path).parent.glob("report*.txt"))
    return candidates[0] if candidates else None


# ---------------------------------------------------------------- המרה לקואורדינטות תמונה

def image_transform(layout: dict, size: int, margin_px: int = 0):
    """מחזיר פונקציה שממירה (x, y) במטרים לפיקסלים בקנבס ריבועי.

    שומר על יחס הממדים (letterbox), צפון למעלה: y בתמונה גדל כלפי מטה.
    """
    W, H = layout["area"]["width_m"], layout["area"]["height_m"]
    usable = size - 2 * margin_px
    s = usable / max(W, H)
    ox = margin_px + (usable - W * s) / 2
    oy = margin_px + (usable - H * s) / 2

    def to_px(x: float, y: float) -> tuple[float, float]:
        return ox + x * s, oy + (H - y) * s

    return to_px, s


def image_boxes(layout: dict, size: int = 512, include: set[str] | None = None) -> list[dict]:
    """תיבות מנורמלות [x0, y0, x1, y1] בתחום [0,1], ראשית בפינה השמאלית-עליונה.

    זה הפורמט ש-GLIGEN ודומיו מצפים לו, וגם הפורמט שמולו יושווה גלאי ה-CV.
    include: רק המזהים האלה (למשל האלמנטים הנראים - בלי מוסתרים ובלי מיקום לא ידוע).
    """
    to_px, _ = image_transform(layout, size)
    out = []
    for el in sorted(layout["elements"], key=lambda e: Z_ORDER[e["class"]]):
        if include is not None and el["id"] not in include:
            continue
        b = Box.from_bbox(el["bbox"])
        px0, py0 = to_px(b.x0, b.y1)  # פינה צפון-מערבית -> שמאל-עליון
        px1, py1 = to_px(b.x1, b.y0)
        out.append({
            "id": el["id"],
            "class": el["class"],
            "phrase": el["phrase"],
            "evidence": el["evidence"],
            "box": [round(px0 / size, 4), round(py0 / size, 4), round(px1 / size, 4), round(py1 / size, 4)],
        })
    return out


def global_caption(layout: dict) -> str:
    """כותרת גלובלית. אלמנטים בלי מיקום נזכרים בה (הטקסט אומר שהם קיימים, רק לא איפה);
    אלמנטים מוסתרים לעולם לא נזכרים - הם לא אמורים להיראות בתמונה."""
    src = layout["source"]
    caption = f"archaeological plan-view reconstruction of a {src['context']}, {src['period']}"
    floating = [e["phrase"] for e in layout["elements"] if e["position"] == "unspecified"]
    return caption + (", with " + ", ".join(floating) if floating else "")
