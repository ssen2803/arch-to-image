"""בדיקות למאמת ולהמרות. כל בדיקה שלילית "מקלקלת" Layout תקין בדרך אחת
ובודקת שהמאמת תופס בדיוק את סוג השגיאה הזה."""

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from layout_core import (  # noqa: E402
    PALETTE, api_schema, check_spans, errors_only, image_boxes, load_layout, validate,
)
from render_layout import render_seg  # noqa: E402

EXAMPLES = sorted((ROOT / "examples").glob("*/layout.gold.json"))


def _ex(name: str) -> dict:
    return load_layout(ROOT / "examples" / name / "layout.gold.json")


def _el(layout: dict, el_id: str) -> dict:
    return next(e for e in layout["elements"] if e["id"] == el_id)


def _error_text(layout: dict) -> str:
    return "\n".join(str(i) for i in errors_only(validate(layout)))


# ------------------------------------------------------------ דוגמאות ה-gold

def test_there_are_seven_examples():
    assert len(EXAMPLES) == 7


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.parent.name)
def test_gold_examples_are_valid(path):
    layout = load_layout(path)
    assert errors_only(validate(layout)) == []
    text = (path.parent / "report.he.txt").read_text(encoding="utf-8")
    assert check_spans(layout, text) == []


def test_examples_cover_all_classes_evidence_levels_and_predicates():
    layouts = [load_layout(p) for p in EXAMPLES]
    classes = {e["class"] for lay in layouts for e in lay["elements"]}
    evidence = {e["evidence"] for lay in layouts for e in lay["elements"]}
    predicates = {r["predicate"] for lay in layouts for r in lay["relations"]}
    assert classes == {"wall", "floor", "installation", "pit"}
    assert evidence == {"attested", "inferred", "hypothetical"}
    assert predicates == {"north_of", "east_of", "west_of", "inside", "on", "adjacent_to", "cuts", "under"}
    positions = {e["position"] for lay in layouts for e in lay["elements"]}
    assert positions == {"stated", "relative", "unspecified"}


# ------------------------------------------------------------ שגיאות שחייבות להיתפס

def test_schema_violation_missing_field():
    lay = _ex("ex01_iron_age_room")
    del lay["elements"][0]["evidence"]
    assert "הפרת סכמה" in _error_text(lay)


def test_schema_violation_unknown_class():
    lay = _ex("ex01_iron_age_room")
    lay["elements"][0]["class"] = "column"
    assert "הפרת סכמה" in _error_text(lay)


def test_directional_relation_violated():
    lay = _ex("ex01_iron_age_room")
    # הרצפה ממזרח לקיר, ולכן הטענה ההפוכה (W1 ממזרח ל-F1) שקרית גיאומטרית
    lay["relations"].append({"subject": "W1", "predicate": "east_of", "object": "F1", "source_span": None})
    assert "אינו מתקיים בגיאומטריה" in _error_text(lay)


def test_contradictory_directional_relations():
    lay = _ex("ex02_chalcolithic_pits")
    lay["relations"].append({"subject": "P1", "predicate": "west_of", "object": "P2", "source_span": None})
    # P2 east_of P1 קיים; P1 west_of P2 עקבי - אינו סתירה
    assert "סתירה" not in _error_text(lay)
    lay["relations"].append({"subject": "P1", "predicate": "east_of", "object": "P2", "source_span": None})
    assert "סתירה" in _error_text(lay)


def test_bbox_outside_area():
    lay = _ex("ex01_iron_age_room")
    _el(lay, "I1")["bbox"]["x"] = 4.8  # 4.8 + 0.8 = 5.6 > 5 + tol
    assert "חורגת משטח החפירה" in _error_text(lay)


def test_truncated_side_must_touch_boundary():
    lay = _ex("ex02_chalcolithic_pits")
    _el(lay, "P4")["bbox"]["x"] = 8.5  # מסומן E אבל רחוק 0.5 מ' מהגבול
    assert "מעבר לגבול E" in _error_text(lay)


def test_id_prefix_must_match_class():
    lay = _ex("ex01_iron_age_room")
    _el(lay, "I1")["class"] = "pit"
    assert "הקידומת אינה תואמת" in _error_text(lay)


def test_unknown_relation_id():
    lay = _ex("ex01_iron_age_room")
    lay["relations"].append({"subject": "W9", "predicate": "north_of", "object": "F1", "source_span": None})
    assert "מזהה לא קיים" in _error_text(lay)


def test_adjacent_requires_touching():
    lay = _ex("ex01_iron_age_room")
    _el(lay, "I1")["bbox"]["y"] = 2.0  # מרחק 1.4 מ' מ-W2
    assert "מרווח" in _error_text(lay)


def test_cuts_requires_overlap():
    lay = _ex("ex02_chalcolithic_pits")
    lay["relations"].append({"subject": "P4", "predicate": "cuts", "object": "F1", "source_span": None})
    assert "אינן חופפות" in _error_text(lay)


def test_on_requires_mostly_within():
    lay = _ex("ex05_roman_courtyard")
    lay["relations"].append({"subject": "W2", "predicate": "on", "object": "F1", "source_span": None})
    assert "נמצא על המושא" in _error_text(lay)


def test_hallucinated_span_is_caught():
    lay = _ex("ex01_iron_age_room")
    text = (ROOT / "examples/ex01_iron_age_room/report.he.txt").read_text(encoding="utf-8")
    lay["elements"][0]["source_span"] = "קיר מלבני גדול מאבני גזית"
    assert any("אינו מופיע" in str(i) for i in check_spans(lay, text))


# ------------------------------------------------------------ המרות ותוצרים

def test_api_schema_has_no_unsupported_keywords():
    dumped = json.dumps(api_schema())
    for kw in ("exclusiveMinimum", "\"minimum\"", "\"pattern\"", "uniqueItems", "minItems", "\"const\"", "$schema"):
        assert kw not in dumped
    # nullable הומר ל-anyOf
    assert "anyOf" in dumped


def test_image_boxes_are_normalised_and_north_up():
    lay = _ex("ex05_roman_courtyard")
    boxes = {b["id"]: b["box"] for b in image_boxes(lay)}
    for x0, y0, x1, y1 in boxes.values():
        assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1
    # W2 צפונית לבור P1 -> בתמונה הוא למעלה (y קטן יותר)
    assert boxes["W2"][1] < boxes["P1"][1]
    # W1 מזרחית לרצפה -> בתמונה הוא מימין
    assert boxes["W1"][0] > boxes["F1"][0]


def test_seg_mask_uses_only_palette_colours():
    import numpy as np
    for path in EXAMPLES:
        arr = np.array(render_seg(load_layout(path), 256)).reshape(-1, 3)
        colours = {tuple(int(v) for v in c) for c in np.unique(arr, axis=0)}
        assert colours <= set(PALETTE.values()), path.parent.name


def test_api_schema_is_itself_a_valid_json_schema():
    from jsonschema import Draft202012Validator
    Draft202012Validator.check_schema(api_schema())


def test_prompt_worked_example_is_valid():
    """הדוגמה שבתוך הפרומפט חייבת לעמוד באותם כללים שהמודל נדרש לעמוד בהם."""
    import re
    prompt = (ROOT / "prompts" / "text_to_layout_system.md").read_text(encoding="utf-8")
    report = re.search(r"Report: `([^`]+)`", prompt).group(1)
    layout = json.loads(re.search(r"```json\n(.*?)\n```", prompt, re.S).group(1))
    assert errors_only(validate(layout)) == []
    assert check_spans(layout, report) == []


def test_validation_does_not_mutate_input():
    lay = _ex("ex04_hellenistic_two_rooms")
    before = copy.deepcopy(lay)
    validate(lay)
    assert lay == before


# ------------------------------------------------------------ סטרטיגרפיה והסתרה (מקרה קצה 2)

from layout_core import Box, stratigraphic_order, uncertainty_box  # noqa: E402
from render_layout import visibility  # noqa: E402


def test_sealed_silo_is_pruned_from_visible_layout():
    lay = _ex("ex07_sealed_silo")
    vis = visibility(lay)
    assert vis["P1"]["status"] == "occluded"          # הממגורה מתחת לרצפה
    assert vis["P2"]["status"] == "visible"           # הבור שחותך את הרצפה נראה
    assert vis["I1"]["status"] == "visible"


def test_occluded_element_absent_from_seg_mask():
    import numpy as np
    lay = _ex("ex07_sealed_silo")
    seg = np.array(render_seg(lay, 512)).reshape(-1, 3)
    pit_px = (seg == PALETTE["pit"]).all(axis=1).sum()
    # רק P2 (קוטר 0.9 מ' בשטח 6 מ') צריך להיות במסכה; P1 (1.2 מ') מוסתר
    expected_p2 = np.pi * (0.45 * 512 / 6) ** 2
    assert abs(pit_px - expected_p2) / expected_p2 < 0.1


def test_removing_under_relation_makes_silo_visible():
    lay = _ex("ex07_sealed_silo")
    lay["relations"] = [r for r in lay["relations"] if r["predicate"] != "under"]
    assert visibility(lay)["P1"]["status"] == "visible"  # בלי היחס, ברירת המחדל: בור מעל רצפה


def test_under_requires_overlap():
    lay = _ex("ex07_sealed_silo")
    lay["relations"].append({"subject": "P2", "predicate": "under", "object": "I1", "source_span": None})
    assert "המושא אינו יכול לכסות" in _error_text(lay)


def test_stratigraphic_cycle_is_rejected():
    lay = _ex("ex07_sealed_silo")
    lay["relations"].append({"subject": "F1", "predicate": "under", "object": "P2", "source_span": None})
    # P2 cuts F1 (P2 מעל F1) + F1 under P2 (P2 מעל F1) - עקבי; עכשיו יוצרים מעגל אמיתי:
    assert "מעגל סטרטיגרפי" not in _error_text(lay)
    lay["relations"].append({"subject": "P2", "predicate": "under", "object": "F1", "source_span": None})
    assert "מעגל סטרטיגרפי" in _error_text(lay)


def test_stratigraphic_order_puts_sealed_below_and_cutting_above():
    order = [e["id"] for e in stratigraphic_order(_ex("ex07_sealed_silo"))]
    assert order.index("P1") < order.index("F1") < order.index("P2")
    assert order.index("F1") < order.index("I1")


# ------------------------------------------------------------ מיקום לא ידוע (מקרה קצה 3)

def test_unspecified_element_not_drawn_but_in_caption():
    import numpy as np
    from layout_core import global_caption
    lay = _ex("ex06_underspecified_pit")
    seg = np.array(render_seg(lay, 512)).reshape(-1, 3)
    assert not (seg == PALETTE["pit"]).all(axis=1).any()     # הבור לא צויר בנקודה מומצאת
    assert "round pit" in global_caption(lay)                 # אבל הטקסט אומר שהוא קיים
    assert visibility(lay)["P1"]["status"] == "unspecified"


def test_uncertainty_region_follows_relations():
    lay = _ex("ex06_underspecified_pit")
    p1 = _el(lay, "P1")
    assert uncertainty_box(lay, p1) == Box(0, 0, 5, 5)        # אין אילוצים: כל השטח
    lay["relations"].append({"subject": "P1", "predicate": "north_of", "object": "W1", "source_span": None})
    r = uncertainty_box(lay, p1)
    assert r.y0 == pytest.approx(0.5) and r.y1 == 5          # מצפון למרכז הקיר


def test_unspecified_placeholder_outside_region_warns():
    lay = _ex("ex06_underspecified_pit")
    lay["relations"].append({"subject": "P1", "predicate": "south_of", "object": "W1", "source_span": None})
    issues = validate(lay)
    assert any("מחוץ לאזור" in i.message for i in issues if i.level == "warning")


# ------------------------------------------------------------ השוואת חילוץ מול gold

from compare_layouts import compare  # noqa: E402


def _text(name: str) -> str:
    return (ROOT / "examples" / name / "report.he.txt").read_text(encoding="utf-8")


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.parent.name)
def test_gold_against_itself_is_perfect(path):
    lay = load_layout(path)
    r = compare(lay, lay, (path.parent / "report.he.txt").read_text(encoding="utf-8"))
    assert r["element_precision"] == 1 and r["element_recall"] == 1 and r["element_invention_rate"] == 0
    assert r["positional_invention_rate"] in (0, None) and r["relation_recall"] in (1, None)
    assert r["box_iou"] == pytest.approx(1)


def test_positional_invention_is_measured():
    gold = _ex("ex06_underspecified_pit")
    pred = copy.deepcopy(gold)
    _el(pred, "P1")["position"] = "relative"   # המחלץ "החליט" איפה הבור, למרות שהטקסט לא אומר
    r = compare(pred, gold, _text("ex06_underspecified_pit"))
    assert r["positional_invention_rate"] == 1 and r["positionally_invented"] == ["P1"]


def test_element_invention_is_measured():
    gold = _ex("ex01_iron_age_room")
    pred = copy.deepcopy(gold)
    extra = copy.deepcopy(_el(pred, "I1"))
    extra.update({"id": "I2", "source_span": "ובתוכו שכבת אפר"})  # "מתקן" שהוא בעצם תוכן הטאבון
    pred["elements"].append(extra)
    r = compare(pred, gold, _text("ex01_iron_age_room"))
    assert r["invented_elements"] == ["I2"] and r["element_invention_rate"] == pytest.approx(1 / 5)


def test_missed_occlusion_is_measured():
    gold = _ex("ex07_sealed_silo")
    pred = copy.deepcopy(gold)
    pred["relations"] = [r for r in pred["relations"] if r["predicate"] != "under"]
    r = compare(pred, gold, _text("ex07_sealed_silo"))
    assert r["occlusion_recall"] == 0       # המחלץ לא גזם את הממגורה הקבורה
