"""בדיקות למדד ההזיות ולמיפוי. אינן דורשות GPU או הורדת מודלים."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hallucination import HallucinationConfig, PaletteEvaluator, element_masks, hallucination_report  # noqa: E402
from mapping import (  # noqa: E402
    FOREIGN, LABEL, UNKNOWN, ade20k, ade_to_project_lut, labels_to_ade_rgb, load_mapping,
    project_labels_from_rgb,
)

from layout_core import load_layout  # noqa: E402  (נטען דרך mapping מנתיב שלב 2)
from render_layout import render_seg  # noqa: E402

BG, FLOOR, WALL, INST, PIT = (LABEL[c] for c in ("background", "floor", "wall", "installation", "pit"))
EXAMPLES = sorted((ROOT.parent / "02_layout_pipeline" / "examples").glob("*/layout.gold.json"))
CFG = HallucinationConfig()


def scene() -> np.ndarray:
    """GT סינתטי 128x128: קיר, רצפה ועליה מתקן."""
    gt = np.zeros((128, 128), np.uint8)
    gt[10:20, 10:110] = WALL
    gt[20:100, 10:110] = FLOOR
    gt[50:70, 50:70] = INST
    return gt


# ------------------------------------------------------------ המדד עצמו

def test_perfect_prediction():
    gt = scene()
    r = hallucination_report(gt.copy(), gt, cfg=CFG)
    assert r["HOR"] == 0 and r["object_recall"] == 1 and r["HAR"] == 0


def test_extra_object_in_empty_area_is_hallucinated():
    gt = scene()
    pred = gt.copy()
    pred[105:125, 100:125] = INST  # מתקן שלא קיים ב-Layout, בשטח הריק
    r = hallucination_report(pred, gt, cfg=CFG)
    assert r["n_hallucinated"] == 1 and r["HOR"] == pytest.approx(1 / 4)
    assert r["per_class"]["installation"]["hallucinated"] == 1
    assert r["HAR"] > 0


def test_wrong_class_on_existing_element_is_hallucinated():
    gt = scene()
    pred = gt.copy()
    pred[50:70, 50:70] = PIT  # במקום המתקן צויר בור
    r = hallucination_report(pred, gt, cfg=CFG)
    assert r["per_class"]["pit"]["hallucinated"] == 1
    assert any(m["class"] == "installation" for m in r["missed"])


def test_small_offset_within_tolerance_is_not_hallucinated():
    gt = scene()
    pred = gt.copy()
    pred[50:70, 50:70] = FLOOR
    pred[54:74, 55:75] = INST  # הוזז ב-4-5 פיקסלים, פחות מ-dilate_px=8
    r = hallucination_report(pred, gt, cfg=CFG)
    assert r["per_class"]["installation"]["hallucinated"] == 0


def test_foreign_object_is_always_hallucinated_even_inside_layout():
    gt = scene()
    pred = gt.copy()
    pred[30:45, 30:45] = FOREIGN  # "אדם" על הרצפה
    r = hallucination_report(pred, gt, cfg=CFG)
    assert r["foreign_components"] == 1 and r["per_class"]["foreign"]["hallucinated"] == 1


def test_blank_image_has_zero_hor_but_zero_recall():
    """ההצדקה לכך ש-HOR מדווח תמיד יחד עם recall: תמונה ריקה 'מנצחת' ב-HOR."""
    gt = scene()
    r = hallucination_report(np.zeros_like(gt), gt, cfg=CFG)
    assert r["HOR"] == 0 and r["object_recall"] == 0


def test_noise_below_min_area_is_ignored():
    gt = scene()
    pred = gt.copy()
    pred[110:115, 110:115] = WALL  # 25 פיקסלים < 64
    r = hallucination_report(pred, gt, cfg=CFG)
    assert r["n_hallucinated"] == 0


def test_coarse_merges_installation_into_built():
    """A stone installation that the evaluator labels as wall is not a hallucination at the coarse level."""
    gt = scene()
    pred = gt.copy()
    pred[50:70, 50:70] = WALL
    fine = hallucination_report(pred, gt, cfg=CFG)
    coarse = hallucination_report(pred, gt, cfg=HallucinationConfig(granularity="coarse"))
    assert fine["per_class"]["wall"]["hallucinated"] == 1
    assert coarse["n_hallucinated"] == 0


def test_coarse_still_catches_invented_structure():
    gt = scene()
    pred = gt.copy()
    pred[105:125, 10:110] = WALL  # a wall invented in the empty area (the room-"closing" pattern)
    r = hallucination_report(pred, gt, cfg=HallucinationConfig(granularity="coarse"))
    assert r["n_hallucinated"] == 1 and r["HOR_area"] > 0


def test_hor_area_is_robust_to_fragmentation():
    """One hallucinated wall split into 4 pieces: HOR jumps, HOR_area stays the same."""
    gt = scene()
    whole, split = gt.copy(), gt.copy()
    whole[105:125, 10:110] = WALL
    for x0 in (10, 35, 60, 85):
        split[105:125, x0:x0 + 23] = WALL
    a = hallucination_report(whole, gt, cfg=CFG)
    b = hallucination_report(split, gt, cfg=CFG)
    assert b["HOR"] > a["HOR"]
    assert abs(b["HOR_area"] - a["HOR_area"]) < 0.05


def test_hallucination_attached_to_real_structure_is_isolated():
    """המקרה שחשף את הפגם בגרסה 1: קיר מומצא שנוגע בקיר אמיתי.
    ההזיה חייבת להיתפס, והקיר האמיתי חייב להיספר כנכון ולא להיפסל יחד איתה."""
    gt = scene()
    pred = gt.copy()
    pred[0:10, 10:110] = WALL      # קיר צפוני מומצא, צמוד לקיר האמיתי (שורות 10-20)
    pred[0:20, 110:128] = WALL     # ועוד המשך שלו מזרחה
    r = hallucination_report(pred, gt, cfg=CFG)
    assert r["per_class"]["wall"]["hallucinated"] >= 1
    assert all(m["class"] != "wall" for m in r["missed"])   # הקיר האמיתי נמצא
    assert r["object_recall"] == 1


def test_outside_excavation_area_is_ignored():
    """קיר שצויר מחוץ לגבול החפירה אינו הזיה: שם המצב לא ידוע."""
    gt = scene()
    valid = np.zeros_like(gt, bool)
    valid[:, :] = True
    valid[110:, :] = False              # השוליים התחתונים מחוץ לשטח החפירה
    pred = gt.copy()
    pred[112:128, 0:128] = WALL         # המודל צייר שם קיר
    with_area = hallucination_report(pred, gt, cfg=CFG, valid=valid)
    without = hallucination_report(pred, gt, cfg=CFG)
    assert without["n_hallucinated"] == 1
    assert with_area["n_hallucinated"] == 0 and with_area["HAR"] == 0


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.parent.name)
def test_area_mask_covers_all_elements(path):
    from hallucination import area_mask
    layout = load_layout(path)
    m = area_mask(layout, 512)
    for el in element_masks(layout, 512):
        assert (el["mask"] & ~m).sum() <= 2 * 512, el["id"]  # לכל היותר שורת פיקסלים של עיגול


def test_unknown_granularity_raises():
    with pytest.raises(ValueError):
        hallucination_report(scene(), scene(), cfg=HallucinationConfig(granularity="medium"))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        hallucination_report(np.zeros((10, 10), np.uint8), np.zeros((12, 12), np.uint8))


# ------------------------------------------------------------ על דוגמאות שלב 2

@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.parent.name)
def test_gt_mask_through_palette_evaluator_is_perfect(path):
    """בדיקת שפיות של כל השרשרת: המסכה עצמה חייבת לקבל HOR=0 ו-recall=1."""
    from hallucination import layout_targets
    layout = load_layout(path)
    seg = np.array(render_seg(layout, 512))
    gt = project_labels_from_rgb(seg)
    t = layout_targets(layout, 512)
    r = hallucination_report(PaletteEvaluator()(seg), gt, t["elements"], CFG, t["valid"], t["uncertain"], t["occluded"])
    assert r["HOR"] == 0 and r["HAR"] == 0
    assert r["occlusion_leak_rate"] in (0, None)
    # המסכה עצמה לא מציירת אלמנטים בלי מיקום, ולכן הם "חסרים" בה - וזה נכון
    assert r["n_elements"] == len(t["elements"]) + len(t["uncertain"])
    assert r["object_recall"] == len(t["elements"]) / r["n_elements"]


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.parent.name)
def test_element_masks_agree_with_rendered_mask(path):
    from hallucination import layout_targets
    layout = load_layout(path)
    gt = project_labels_from_rgb(np.array(render_seg(layout, 512)))
    for el in layout_targets(layout, 512)["elements"]:  # רק האלמנטים הנראים
        assert el["mask"].any(), el["id"]
        # הפיקסלים הנראים של האלמנט צבועים במחלקה שלו
        assert (gt[el["mask"]] == el["label"]).any(), el["id"]


# ------------------------------------------------------------ מיפוי ADE20K

def test_ade_palette_is_complete():
    classes, palette = ade20k()
    assert len(classes) == 150 and palette.shape == (150, 3)
    assert classes[0] == "wall" and tuple(palette[0]) == (120, 120, 120)
    assert classes[3] == "floor" and tuple(palette[3]) == (80, 50, 50)


def test_evaluation_groups_are_disjoint_and_contain_condition_class():
    m = load_mapping()
    seen = {}
    for proj, members in m["evaluation_groups"].items():
        assert m["condition"][proj] in members
        for ade in members:
            assert ade not in seen, f"{ade} גם ב-{seen.get(ade)} וגם ב-{proj}"
            seen[ade] = proj


def test_override_moves_class_between_groups():
    with pytest.warns(UserWarning):
        m = load_mapping(overrides={"pit": "sand"})
    assert "sand" in m["evaluation_groups"]["pit"]
    assert "sand" not in m["evaluation_groups"]["background"]


def test_unknown_ade_class_rejected():
    with pytest.raises(ValueError):
        load_mapping(overrides={"pit": "hole"})


def test_condition_image_uses_only_mapped_ade_colours():
    m = load_mapping()
    allowed = {tuple(int(v) for v in ade20k()[1][ade20k()[0].index(a)]) for a in m["condition"].values()}
    for path in EXAMPLES:
        labels = project_labels_from_rgb(np.array(render_seg(load_layout(path), 512)))
        rgb = labels_to_ade_rgb(labels, m)
        assert {tuple(int(v) for v in c) for c in np.unique(rgb.reshape(-1, 3), axis=0)} <= allowed


def test_lut_round_trip():
    m = load_mapping()
    lut = ade_to_project_lut(m)
    classes, _ = ade20k()
    for proj, ade in m["condition"].items():
        assert lut[classes.index(ade)] == LABEL[proj]
    assert lut[classes.index("person")] == FOREIGN
    assert lut[classes.index("car")] == FOREIGN


def test_unknown_colours_rejected_for_conditioning():
    labels = np.full((4, 4), UNKNOWN, np.uint8)
    with pytest.raises(ValueError):
        labels_to_ade_rgb(labels, load_mapping())


# ------------------------------------------------------------ מיקום לא ידוע והסתרה (מקרי קצה 2-3)

def _targets(name):
    from hallucination import layout_targets
    layout = load_layout(ROOT.parent / "02_layout_pipeline" / "examples" / name / "layout.gold.json")
    gt = project_labels_from_rgb(np.array(render_seg(layout, 512)))
    return layout, gt, layout_targets(layout, 512)


def _disc(shape, cy, cx, r, label, base):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    out = base.copy()
    out[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = label
    return out


def _report(pred, gt, t):
    return hallucination_report(pred, gt, t["elements"], CFG, t["valid"], t["uncertain"], t["occluded"])


def test_unspecified_pit_anywhere_in_region_is_not_hallucination():
    _, gt, t = _targets("ex06_underspecified_pit")
    pred = _disc(gt.shape, 150, 380, 50, PIT, gt)    # בור במקום כלשהו בשטח (r=50px ~ קוטר 1 מ')
    r = _report(pred, gt, t)
    assert r["n_hallucinated"] == 0 and r["uncertain_found"] == 1 and r["object_recall"] == 1


def test_second_pit_beyond_what_text_mentions_is_hallucination():
    _, gt, t = _targets("ex06_underspecified_pit")
    pred = _disc(gt.shape, 150, 380, 50, PIT, gt)
    pred = _disc(gt.shape, 200, 120, 45, PIT, pred)   # הטקסט מזכיר בור אחד בלבד
    r = _report(pred, gt, t)
    assert r["n_hallucinated"] == 1 and r["uncertain_found"] == 1


def test_missing_unspecified_pit_lowers_recall():
    _, gt, t = _targets("ex06_underspecified_pit")
    r = _report(gt.copy(), gt, t)
    assert r["uncertain_found"] == 0
    assert any(m.get("kind") == "location_unspecified" for m in r["missed"])


def test_drawing_sealed_silo_is_occlusion_leak():
    layout, gt, t = _targets("ex07_sealed_silo")
    assert [o["id"] for o in t["occluded"]] == ["P1"]
    clean = _report(gt.copy(), gt, t)
    assert clean["occlusion_leak_rate"] == 0
    silo = t["occluded"][0]["mask"]
    ys, xs = np.nonzero(silo)
    leaked = gt.copy()
    leaked[silo] = PIT                                # המודל צייר את הממגורה הקבורה
    r = _report(leaked, gt, t)
    assert r["occlusion_leak_rate"] == 1
    assert r["per_class"]["pit"]["hallucinated"] >= 1  # והיא נספרת גם כהזיה רגילה
