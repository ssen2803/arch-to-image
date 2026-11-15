"""בדיקות לשלב 4 (ללא GPU וללא הורדת מודלים)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from common import IGNORE  # noqa: E402
from compositions import composition_tags  # noqa: E402
from convert_annotations import color_table, convert  # noqa: E402
from datasets import boundary_ignore, drop_blank_images, filter_rows, is_blank_image, split_by_layout  # noqa: E402
from metrics import (  # noqa: E402
    agreement, bootstrap_miou, coarsen_labels, confusion, iou_from_confusion, miou, paired_bootstrap_delta,
)
from synth_layouts import generate_layout  # noqa: E402

from layout_core import errors_only, validate  # noqa: E402

BG, FLOOR, WALL, INST, PIT = 0, 1, 2, 3, 4


# ------------------------------------------------------------ מדדים

def test_perfect_prediction_miou_is_one():
    gt = np.random.default_rng(0).integers(0, 5, (32, 32)).astype(np.uint8)
    assert miou(confusion(gt, gt)) == 1.0


def test_ignore_pixels_do_not_count():
    gt = np.array([[1, 1], [IGNORE, IGNORE]], np.uint8)
    pred = np.array([[1, 1], [2, 3]], np.uint8)
    assert miou(confusion(pred, gt)) == 1.0


def test_known_iou_value():
    gt = np.array([[1, 1, 1, 1]], np.uint8)
    pred = np.array([[1, 1, 0, 0]], np.uint8)
    iou = iou_from_confusion(confusion(pred, gt))
    assert iou[1] == pytest.approx(0.5) and iou[0] == 0.0
    assert np.isnan(iou[2])  # מחלקה שלא הופיעה בכלל לא נכנסת לממוצע


def test_out_of_range_prediction_counts_as_miss():
    gt = np.array([[2, 2]], np.uint8)
    pred = np.array([[2, 5]], np.uint8)  # 5 = FOREIGN של שלב 3
    assert iou_from_confusion(confusion(pred, gt))[2] == pytest.approx(0.5)


def test_coarsen_merges_classes():
    lab = np.array([[0, 1, 2, 3, 4, IGNORE]], np.uint8)
    assert coarsen_labels(lab).tolist() == [[0, 0, 1, 1, 2, IGNORE]]


def test_agreement_detects_mismatch():
    label = np.zeros((64, 64), np.uint8)
    label[10:20, :] = WALL
    assert agreement(label, label)["agree_miou"] == 1.0
    wrong = label.copy()
    wrong[40:60, :] = WALL  # "קיר" שהדיפוזיה המציאה
    assert agreement(wrong, label)["built_iou"] < 0.5


def test_bootstrap_ci_contains_point_and_paired_delta_sign():
    rng = np.random.default_rng(1)
    gts = [rng.integers(0, 5, (16, 16)).astype(np.uint8) for _ in range(20)]
    good = [confusion(g, g) for g in gts]
    bad = [confusion(np.zeros_like(g), g) for g in gts]
    p, lo, hi = bootstrap_miou(good, n_boot=200)
    assert lo <= p <= hi
    d = paired_bootstrap_delta(good, bad, n_boot=200)
    assert d["delta"] > 0 and d["ci95"][0] > 0 and d["p_delta_le_0"] == 0


# ------------------------------------------------------------ זליגת תוויות (מקרה קצה 1)

def test_boundary_ignore_band():
    lab = np.zeros((20, 20), np.uint8)
    lab[:, 10:] = WALL
    out = boundary_ignore(lab, 2)
    assert (out[:, 8:12] == IGNORE).all()            # רצועה של 2 פיקסלים לכל צד
    assert (out[:, :7] == BG).all() and (out[:, 13:] == WALL).all()


def test_filter_rows_thresholds():
    rows = [{"agree_miou": 0.8, "HAR": 0.1}, {"agree_miou": 0.1, "HAR": 0.1}, {"agree_miou": 0.9, "HAR": 0.9}]
    assert len(filter_rows(rows, min_agree=0.3, max_har=0.6)) == 1


def test_is_blank_image():
    rng = np.random.default_rng(0)
    assert is_blank_image(np.zeros((64, 64, 3), np.uint8))                    # שחורה (מסנן הבטיחות)
    assert is_blank_image(np.full((64, 64, 3), 128, np.uint8))                # אחידה
    assert not is_blank_image(rng.integers(0, 255, (64, 64, 3)).astype(np.uint8))
    dark = rng.integers(0, 40, (64, 64)).astype(np.uint8)                     # כהה אבל עם תוכן
    assert not is_blank_image(dark)


def test_drop_blank_images_and_remove_blank(tmp_path):
    from PIL import Image
    import csv
    (tmp_path / "images").mkdir()
    rng = np.random.default_rng(1)
    Image.fromarray(np.zeros((32, 32, 3), np.uint8)).save(tmp_path / "images" / "a_s0.png")
    Image.fromarray(rng.integers(0, 255, (32, 32, 3)).astype(np.uint8)).save(tmp_path / "images" / "a_s1.png")
    rows = [{"image_path": tmp_path / "images" / n, "image": f"images/{n}"} for n in ("a_s0.png", "a_s1.png")]
    kept, dropped = drop_blank_images(rows)
    assert [r["image"] for r in kept] == ["images/a_s1.png"]
    assert [r["image"] for r in dropped] == ["images/a_s0.png"]

    from build_synthetic import FIELDS, remove_blank
    with open(tmp_path / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for n in ("a_s0.png", "a_s1.png"):
            w.writerow({k: "" for k in FIELDS} | {"image": f"images/{n}", "label": "labels/a.png"})
    assert remove_blank(tmp_path) == ["images/a_s0.png"]
    assert not (tmp_path / "images" / "a_s0.png").exists()
    with open(tmp_path / "manifest.csv", encoding="utf-8") as f:
        assert [r["image"] for r in csv.DictReader(f)] == ["images/a_s1.png"]


def test_split_by_layout_has_no_leakage():
    rows = [{"layout_id": f"synth_{i:05d}", "seed": s} for i in range(50) for s in (0, 1)]
    train, val = split_by_layout(rows)
    assert not ({r["layout_id"] for r in train} & {r["layout_id"] for r in val})
    assert len(val) == 10  # 5 layouts x 2 seeds


# ------------------------------------------------------------ קומפוזיציות (מקרה קצה 4)

def _scene():
    lab = np.zeros((128, 128), np.uint8)
    lab[20:100, 20:100] = FLOOR
    return lab


def test_pit_in_floor_vs_open():
    lab = _scene()
    lab[50:62, 50:62] = PIT
    assert "pit_in_floor" in composition_tags(lab)
    lab2 = np.zeros((128, 128), np.uint8)
    lab2[50:62, 50:62] = PIT
    assert composition_tags(lab2) >= {"pit_in_open"} and "pit_in_floor" not in composition_tags(lab2)


def test_directional_wall_pit_tags():
    lab = np.zeros((128, 128), np.uint8)
    lab[40:52, 40:52] = PIT
    lab[70:78, 20:100] = WALL           # מתחת לבור בתמונה = מדרום (בתמונה מיושרת לצפון)
    t = composition_tags(lab)
    assert "wall_south_of_pit" in t and "wall_north_of_pit" not in t
    assert "wall_north_of_pit" in composition_tags(np.flipud(lab).copy())  # היפוך אנכי הופך את התגית


# ------------------------------------------------------------ נתונים סינתטיים

def test_synthetic_layouts_valid_and_deterministic():
    for s in range(40):
        lay = generate_layout(s)
        assert not errors_only(validate(lay)), s
    assert generate_layout(7) == generate_layout(7)


def test_synthetic_pool_contains_each_composition():
    """כדי שאפשר יהיה להוציא כל קומפוזיציה מהאימון ולבחון עליה, היא חייבת להופיע במאגר."""
    from render_layout import render_seg
    from mapping import project_labels_from_rgb
    seen = set()
    for s in range(120):
        seen |= composition_tags(project_labels_from_rgb(np.array(render_seg(generate_layout(s), 256))),
                                 min_area=16, near_px=32)
    assert {"pit_in_floor", "pit_in_open", "installation_on_floor", "wall_south_of_pit", "wall_north_of_pit"} <= seen


# ------------------------------------------------------------ המרת תיוג

def test_convert_annotations_round_trip():
    table = color_table()
    rgb = np.zeros((4, 4, 3), np.uint8)              # שחור = רקע
    rgb[0, :] = (0xd6, 0xb2, 0x70)                    # floor
    rgb[1, :] = (0xff, 0x00, 0xff)                    # ignore
    out, unknown = convert(rgb, table)
    assert not unknown
    assert out[0].tolist() == [FLOOR] * 4 and out[1].tolist() == [IGNORE] * 4 and out[2].tolist() == [BG] * 4
    rgb[3, 0] = (1, 2, 3)
    _, unknown = convert(rgb, table)
    assert unknown == {"(1, 2, 3)": 1}


# ------------------------------------------------------------ רשת

def test_unet_shapes_and_size():
    import torch
    from unet import ResNet18UNet, count_parameters
    m = ResNet18UNet(5, pretrained=False).eval()
    with torch.no_grad():
        assert m(torch.zeros(1, 3, 96, 128)).shape == (1, 5, 96, 128)
    assert 13e6 < count_parameters(m)["total"] < 16e6


def test_evaluate_refuses_non_real_split(tmp_path):
    import subprocess
    r = subprocess.run([sys.executable, str(ROOT / "src" / "evaluate.py"), "--split", str(tmp_path),
                        "--out", str(tmp_path / "o")], capture_output=True, text=True, encoding="utf-8")
    assert r.returncode != 0 and "data/real" in (r.stdout + r.stderr)


def test_rating_auc_extremes_and_ties():
    from rating_auc import auc
    good = np.array([True, True, False, False])
    assert auc(np.array([0.9, 0.8, 0.2, 0.1]), good) == 1.0        # הציון מפריד בדיוק
    assert auc(np.array([0.1, 0.2, 0.8, 0.9]), good) == 0.0        # הפוך
    assert auc(np.array([0.5, 0.5, 0.5, 0.5]), good) == 0.5        # תיקו מלא = ניחוש
    assert np.isnan(auc(np.array([0.5, 0.6]), np.array([True, True])))
