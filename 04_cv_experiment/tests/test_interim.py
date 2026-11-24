"""בדיקות לקוד שנוסף לקראת מצגת הביניים. אף אחת לא דורשת GPU או נתונים."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from datasets import split_by_layout, split_by_layout3  # noqa: E402
from error_analysis import canon, surface_map  # noqa: E402
from prepare_real_walls import group_overlaps, nodata_mask, overlap_score, wall_mask  # noqa: E402
from real_walls_eval import cluster_bootstrap, counts, scores  # noqa: E402

ROWS = [{"layout_id": f"synth_{i:05d}", "image": f"{i}_{s}"} for i in range(40) for s in (0, 1)]


def test_three_way_split_is_by_layout_and_keeps_the_old_validation_set():
    tr, va, te = split_by_layout3(ROWS)
    ids = [{r["layout_id"] for r in s} for s in (tr, va, te)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert len(tr) + len(va) + len(te) == len(ROWS)
    assert va == split_by_layout(ROWS)[1]


def test_overlap_is_found_for_shifted_crops_and_not_for_unrelated_images():
    rng = np.random.default_rng(0)
    big = rng.normal(size=(400, 400))
    a, b = big[:200, :200], big[150:350, :200]            # חפיפה של 25%
    c = rng.normal(size=(200, 200))
    assert overlap_score(a, b)[0] > 0.1
    assert overlap_score(a, c)[0] < 0.1
    groups, pairs = group_overlaps({"a": a, "b": b, "c": c}, thr=0.1)
    assert groups["a"] == groups["b"] != groups["c"] and len(pairs) == 1


def test_identical_images_score_one():
    a = np.random.default_rng(1).normal(size=(200, 200))
    assert overlap_score(a, a)[0] > 0.95


def test_nodata_needs_a_large_black_area():
    img = np.full((100, 100, 3), 120, np.uint8)
    img[:10, :10] = 0
    assert not nodata_mask(img).any()
    img[:, :40] = 0
    assert nodata_mask(img)[:, :40].all() and not nodata_mask(img)[:, 41:].any()


def test_every_polygon_is_a_wall_whatever_its_label():
    shapes = [{"label": l, "points": [[x, 0], [x + 9, 0], [x + 9, 9], [x, 9]]} for l, x in (("..", 0), ("stone", 20), ("\\\\\\", 40))]
    assert wall_mask(shapes, (60, 20)).sum() == 3 * 100


def test_scores_ignore_unlabelled_pixels():
    gt = np.array([[2, 2, 0, 255]])
    c = counts(np.array([[True, False, True, True]]), gt)
    assert c.tolist() == [1, 1, 1, 0]
    s = scores(c)
    assert abs(s["iou"] - 1 / 3) < 1e-9 and s["precision"] == s["recall"] == 0.5


def test_cluster_bootstrap_resamples_groups_and_is_reproducible():
    cs = np.array([[10, 0, 0, 0], [10, 0, 0, 0], [0, 5, 5, 0]])
    g = np.array([0, 0, 1])
    b = cluster_bootstrap(cs, g, n=500)
    assert np.array_equal(b, cluster_bootstrap(cs, g, n=500))
    assert set(np.round(b, 6)) <= {0.0, round(20 / 30, 6), 1.0}       # רק צירופים של קבוצות שלמות


def test_inverse_relations_are_the_same_claim():
    assert canon(("A", "south_of", "B")) == canon(("B", "north_of", "A"))
    assert canon(("A", "east_of", "B")) == canon(("B", "west_of", "A"))
    assert canon(("A", "adjacent_to", "B")) == canon(("B", "adjacent_to", "A"))
    assert canon(("A", "inside", "B")) != canon(("B", "inside", "A"))


def test_surface_map_separates_grass_shadow_and_bright_stone():
    a = np.array([[[60, 120, 50], [10, 10, 10], [200, 200, 195], [120, 105, 90]]], np.uint8)
    assert surface_map(a).tolist() == [[0, 1, 2, 3]]
