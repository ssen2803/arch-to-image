"""מדידת הזיות אובייקטים (Hallucinated Object Rate) בתמונה שנוצרה מ-Layout.

רעיון: ה-Layout הוא "רשימת האובייקטים המותרים". מפיקים מהתמונה שנוצרה
מפת מחלקות (בעזרת מעריך) ובודקים מה צויר מחוץ למה שה-Layout מתיר.

  אזור מותר של מחלקה c = אזור ה-GT של c, מורחב ב-dilate_px (סובלנות מיקום).
  אובייקט מוזה = רכיב קשיר (מעל min_area_px) של פיקסלים ממחלקה c מחוץ לאזור
                 המותר שלה, או כל רכיב ממחלקה זרה (FOREIGN).
  אובייקט נכון = אלמנט ב-Layout שצויר במחלקה הנכונה (recall).

  HOR = מוזים / (מוזים + נכונים)   (המבנה של CHAIR, Rohrbach et al. 2018)

HOR לבדו ניתן "לרמות" - תמונה ריקה מקבלת HOR=0. לכן הוא מדווח תמיד
יחד עם Object Recall (כמה מאלמנטי ה-Layout אכן צוירו) ועם HAR
(כמה מהשטח שאמור להיות ריק התמלא במשהו).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from mapping import FOREIGN, LABEL, PROJECT_CLASSES, UNKNOWN

BG = LABEL["background"]
FG_LABELS = [LABEL[c] for c in PROJECT_CLASSES if c != "background"]
LABEL_NAMES = {**{i: c for i, c in enumerate(PROJECT_CLASSES)}, FOREIGN: "foreign", UNKNOWN: "unknown"}


@dataclass(frozen=True)
class HallucinationConfig:
    min_area_px: int = 64        # רכיב קטן מזה = רעש, לא אובייקט (8x8 פיקסלים ב-512)
    dilate_px: int = 8           # סובלנות מיקום: ~3% מרוחב התמונה
    recall_ratio: float = 0.5    # אלמנט נחשב "צויר" אם לפחות חצי מהחלק הנראה שלו סווג נכון
    granularity: str = "fine"    # "fine" = 5 מחלקות; "coarse" = בנוי (קיר+מתקן) מול קרקע (רצפה+רקע)


# ברזולוציה הגסה: מחלקות שהמעריך אינו מסוגל להפריד חזותית מאוחדות.
# קיר ומתקן מאבן נראים אותו דבר, וכך גם רצפת עפר ואדמת רקע. ראו measurement_HOR.md סעיף 6.
COARSE = {LABEL["background"]: LABEL["background"], LABEL["floor"]: LABEL["background"],
          LABEL["wall"]: LABEL["wall"], LABEL["installation"]: LABEL["wall"], LABEL["pit"]: LABEL["pit"],
          FOREIGN: FOREIGN, UNKNOWN: UNKNOWN}


def coarsen(labels: np.ndarray) -> np.ndarray:
    lut = np.arange(256, dtype=np.uint8)
    for src, dst in COARSE.items():
        lut[src] = dst
    return lut[labels]


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    return ndimage.binary_dilation(mask, structure=np.ones((3, 3), bool), iterations=px)


def components(pred: np.ndarray, label: int, min_area: int) -> list[dict]:
    """רכיבים קשירים (קישוריות 8) של תווית אחת, מעל סף שטח."""
    lab, n = ndimage.label(pred == label, structure=np.ones((3, 3), bool))
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        mask = lab[sl] == i
        area = int(mask.sum())
        if area >= min_area:
            out.append({"slice": sl, "mask": mask, "area": area,
                        "bbox": [sl[1].start, sl[0].start, sl[1].stop, sl[0].stop]})  # x0,y0,x1,y1
    return out


def hallucination_report(pred: np.ndarray, gt: np.ndarray, elements: list[dict] | None = None,
                         cfg: HallucinationConfig = HallucinationConfig(),
                         valid: np.ndarray | None = None,
                         uncertain: list[dict] | None = None,
                         occluded: list[dict] | None = None,
                         return_component_map: bool = False) -> dict:
    """pred, gt: מפות תוויות באותו גודל. elements: [{id, label, mask}] מה-Layout (אופציונלי).

    בלי elements, ה-recall מחושב על רכיבים קשירים של ה-GT במקום על אלמנטים.
    valid: מסכה בוליאנית של שטח החפירה. מחוץ לה המצב לא ידוע (לא נחפר), ולכן
    שום דבר שם אינו נספר כהזיה וגם אינו נחשב "שטח ריק" ל-HAR.
    return_component_map: מוסיף "component_map", מפה שבה k = הרכיב ה-k ברשימת hallucinated (מבוסס 1).
    משמש לביקורת האנושית (audit_sample.py). החישוב עצמו לא משתנה.
    """
    if pred.shape != gt.shape:
        raise ValueError(f"גודל לא תואם: pred {pred.shape} מול gt {gt.shape}")
    if valid is not None:
        pred = np.where(valid, pred, UNKNOWN).astype(np.uint8)
        gt = np.where(valid, gt, UNKNOWN).astype(np.uint8)
    def _coarse_items(items):
        if not items:
            return []
        if cfg.granularity != "coarse":
            return list(items)
        # רצפה מתמזגת לקרקע ברזולוציה הגסה: אינה אובייקט
        return [{**e, "label": int(COARSE[e["label"]])} for e in items if COARSE[e["label"]] != BG]

    if cfg.granularity == "coarse":
        pred, gt = coarsen(pred), coarsen(gt)
    elif cfg.granularity != "fine":
        raise ValueError(f"granularity לא מוכר: {cfg.granularity}")
    if elements is not None:
        elements = _coarse_items(elements)
    uncertain, occluded = _coarse_items(uncertain), _coarse_items(occluded)

    # עודף: פיקסלים של מחלקה c מחוץ לאזור המותר שלה (GT מורחב). כל רכיב קשיר של
    # העודף הוא אובייקט מוזה. כך הזיה שמחוברת למבנה אמיתי (קיר "סוגר" שנוגע בקיר
    # קיים) מבודדת ממנו, והמבנה האמיתי אינו נענש. (בגרסה 1 נבדקה תמיכה ברמת רכיב שלם,
    # וזה נכשל בדיוק במקרה הזה. ראו measurement_HOR.md סעיף 6.)
    #
    # אלמנט בלי מיקום (מקרה קצה 3): רכיב ממחלקתו שנמצא ברובו בתוך אזור האי-ודאות שלו
    # אינו הזיה - אבל רק עד מספר האלמנטים כאלה שהטקסט מזכיר (הגדולים ביותר נחשבים להם).
    excess_any = pred == FOREIGN
    hallucinated, uncertain_found, uncertain_missed = [], 0, []
    comp_map = np.zeros(pred.shape, np.int16) if return_component_map else None
    for c in FG_LABELS:
        excess = (pred == c) & ~_dilate(gt == c, cfg.dilate_px)
        comps = components(excess.astype(np.uint8), 1, cfg.min_area_px)
        regions = [u for u in uncertain if u["label"] == c]
        supported_idx: set[int] = set()
        if regions:
            union = np.logical_or.reduce([u["region"] for u in regions])
            cand = sorted((k for k, comp in enumerate(comps)
                           if (comp["mask"] & union[comp["slice"]]).sum() >= 0.5 * comp["area"]),
                          key=lambda k: -comps[k]["area"])
            for u, k in zip(sorted(regions, key=lambda u: -u["expected_area"]), cand):
                supported_idx.add(k)
                excess[comps[k]["slice"]] &= ~comps[k]["mask"]
            for j, u in enumerate(sorted(regions, key=lambda u: -u["expected_area"])):
                k = cand[j] if j < len(cand) else None
                if k is not None and comps[k]["area"] >= 0.3 * u["expected_area"]:
                    uncertain_found += 1
                else:
                    uncertain_missed.append({"id": u["id"], "class": LABEL_NAMES[c], "coverage": 0.0,
                                             "bbox": u["bbox"], "kind": "location_unspecified"})
        excess_any |= excess
        for k, comp in enumerate(comps):
            if k not in supported_idx:
                hallucinated.append({"class": LABEL_NAMES[c], "kind": "unsupported",
                                     "area": comp["area"], "bbox": comp["bbox"]})
                if comp_map is not None:
                    comp_map[comp["slice"]][comp["mask"]] = len(hallucinated)
    for comp in components(pred, FOREIGN, cfg.min_area_px):
        hallucinated.append({"class": "foreign", "kind": "foreign", "area": comp["area"], "bbox": comp["bbox"]})
        if comp_map is not None:
            comp_map[comp["slice"]][comp["mask"]] = len(hallucinated)

    per_class = {name: {"hallucinated": sum(h["class"] == name for h in hallucinated)}
                 for name in [LABEL_NAMES[c] for c in FG_LABELS] + ["foreign"]}

    # HAR: שטח שאמור להיות ריק (רחוק מכל אלמנט) ומולא במחלקה כלשהי
    empty_far = (gt == BG) & ~_dilate((gt != BG) & (gt != UNKNOWN), cfg.dilate_px)
    filled = (pred != BG) & (pred != UNKNOWN)
    har = float((filled & empty_far).sum() / empty_far.sum()) if empty_far.any() else 0.0

    # Object Recall: כמה מאלמנטי ה-Layout צוירו במחלקה הנכונה (על החלק הנראה שלהם)
    if elements is None:
        elements = []
        for c in FG_LABELS:
            for k, comp in enumerate(components(gt, c, 1)):
                full = np.zeros_like(gt, dtype=bool)
                full[comp["slice"]] = comp["mask"]
                elements.append({"id": f"{LABEL_NAMES[c]}_{k}", "label": c, "mask": full})
    found, missed = 0, []
    for el in elements:
        visible = el["mask"] & (gt == el["label"])
        if not visible.any():
            continue
        ratio = float((visible & (pred == el["label"])).sum() / visible.sum())
        if ratio >= cfg.recall_ratio:
            found += 1
        else:
            ys, xs = np.nonzero(visible)
            missed.append({"id": el["id"], "class": LABEL_NAMES[el["label"]], "coverage": round(ratio, 3),
                           "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]})
    found += uncertain_found
    missed += uncertain_missed
    n_el = found + len(missed)

    # זליגת הסתרה (מקרה קצה 2): אלמנט שקבור מתחת לאחר לא אמור להיראות. בודקים אם המודל
    # צייר את מחלקתו בטביעת הרגל שלו, במקום שבו ה-GT הנראה הוא מחלקה אחרת.
    occlusion_tests = []
    for o in occluded:
        c = o["label"]
        region = o["mask"] & (gt != c) & (gt != UNKNOWN)
        if region.sum() < cfg.min_area_px:
            continue
        frac = float(((pred == c) & region).sum() / region.sum())
        occlusion_tests.append({"id": o["id"], "class": LABEL_NAMES[c], "leak_fraction": round(frac, 3),
                                "leaked": frac >= 0.3})

    # HOR במבנה של CHAIR: מתוך כל האובייקטים שצוירו (מוזים + אלמנטים שצוירו נכון), כמה מוזים.
    # המכנה סופר אובייקטים נכונים לפי אלמנטי ה-Layout, ולכן אינו תלוי בפיצול שלהם בתמונה.
    n_hall = len(hallucinated)
    drawn_fg = (pred != BG) & (pred != UNKNOWN)
    extra = {"component_map": comp_map} if comp_map is not None else {}
    return {**extra,
        "occlusion_leak_rate": (round(sum(t["leaked"] for t in occlusion_tests) / len(occlusion_tests), 4)
                                if occlusion_tests else None),
        "occlusion_tests": occlusion_tests,
        "uncertain_found": uncertain_found,
        "HOR": round(n_hall / (n_hall + found), 4) if (n_hall + found) else 0.0,
        # משוקלל-שטח: חלק הפיקסלים שצוירו (לא רקע) שאין להם עיגון. עמיד לפיצול גם של ההזיות.
        "HOR_area": round(float(excess_any.sum() / drawn_fg.sum()), 4) if drawn_fg.any() else 0.0,
        "n_hallucinated": n_hall,
        "n_correct": found,
        "foreign_components": per_class["foreign"]["hallucinated"],
        "HAR": round(har, 4),
        "object_recall": round(found / n_el, 4) if n_el else 1.0,
        "n_elements": n_el,
        "missed": missed,
        "per_class": per_class,
        "hallucinated": hallucinated,
        "unknown_pixel_ratio": round(float((pred == UNKNOWN).mean()), 4),
        "config": asdict(cfg),
    }


def area_mask(layout: dict, size: int = 512) -> np.ndarray:
    """מסכת שטח החפירה בתמונה. מחוצה לה יש שוליים (letterbox) כששטח החפירה אינו ריבועי."""
    from layout_core import image_transform
    to_px, _ = image_transform(layout, size)
    x0, y0 = to_px(0, layout["area"]["height_m"])
    x1, y1 = to_px(layout["area"]["width_m"], 0)
    mask = np.zeros((size, size), bool)
    mask[round(y0):round(y1), round(x0):round(x1)] = True
    return mask


def layout_targets(layout: dict, size: int = 512) -> dict:
    """כל מה שההערכה צריכה מה-Layout: מסכות אלמנטים, אזורי אי-ודאות, טביעות רגל של מוסתרים."""
    from layout_core import image_transform, uncertainty_box
    from render_layout import visibility
    to_px, scale = image_transform(layout, size)
    vis = visibility(layout, size)
    masks = {m["id"]: m for m in element_masks(layout, size)}
    uncertain, occluded = [], []
    for el in layout["elements"]:
        status = vis[el["id"]]["status"]
        if status == "unspecified":
            r = uncertainty_box(layout, el)
            (x0, y0), (x1, y1) = to_px(r.x0, r.y1), to_px(r.x1, r.y0)
            region = np.zeros((size, size), bool)
            region[round(y0):round(y1), round(x0):round(x1)] = True
            uncertain.append({"id": el["id"], "label": LABEL[el["class"]], "region": region,
                              "expected_area": float(masks[el["id"]]["mask"].sum()),
                              "bbox": [round(x0), round(y0), round(x1), round(y1)]})
        elif status == "occluded":
            occluded.append({"id": el["id"], "label": LABEL[el["class"]], "mask": masks[el["id"]]["mask"]})
    visible = [m for i, m in masks.items() if vis[i]["status"] == "visible"]
    return {"elements": visible, "uncertain": uncertain, "occluded": occluded, "valid": area_mask(layout, size)}


def element_masks(layout: dict, size: int = 512) -> list[dict]:
    """מסכה בוליאנית לכל אלמנט ב-Layout, באותה המרה שבה נוצרה מסכת ההתניה (שלב 2)."""
    from PIL import Image, ImageDraw

    from layout_core import Box, image_transform
    to_px, _ = image_transform(layout, size)
    out = []
    for el in layout["elements"]:
        b = Box.from_bbox(el["bbox"])
        x0, y0 = to_px(b.x0, b.y1)
        x1, y1 = to_px(b.x1, b.y0)
        img = Image.new("1", (size, size), 0)
        rect = [round(x0), round(y0), round(x1) - 1, round(y1) - 1]
        (ImageDraw.Draw(img).ellipse if el["shape"] == "ellipse" else ImageDraw.Draw(img).rectangle)(rect, fill=1)
        out.append({"id": el["id"], "label": LABEL[el["class"]], "mask": np.array(img, dtype=bool)})
    return out


# ---------------------------------------------------------------- מעריכים: תמונה -> מפת תוויות

class PaletteEvaluator:
    """לתמונות בסגנון הסכמה של הפרויקט (פלטה קבועה): כימות לצבע הקרוב.

    משמש גם לבדיקת שפיות: על מסכת ה-GT עצמה הוא חייב להחזיר HOR=0 ו-recall=1.
    """

    def __init__(self, max_dist: float = 40.0):
        self.max_dist = max_dist

    def __call__(self, image_rgb: np.ndarray) -> np.ndarray:
        from mapping import project_labels_from_rgb
        return project_labels_from_rgb(image_rgb, self.max_dist)


class SegformerEvaluator:
    """מקטע ADE20K (SegFormer) + מיפוי קבוצות -> תוויות פרויקט / FOREIGN.

    זה מעריך חיצוני ובלתי תלוי במחולל (גישת cycle-consistency כמו ב-ControlNet++).
    יש לו פער תחום על איורים, ולכן חובה לכייל אותו מול ביקורת אנושית (ראו measurement_HOR.md).
    """

    def __init__(self, mapping: dict, model_id: str = "nvidia/segformer-b4-finetuned-ade-512-512",
                 device: str | None = None):
        import torch
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

        from mapping import ade_to_project_lut
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(self.device).eval()
        self.lut = ade_to_project_lut(mapping)
        self.model_id = model_id

    def ade_labels(self, image_rgb: np.ndarray) -> np.ndarray:
        torch = self.torch
        inputs = self.processor(images=image_rgb, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            logits = self.model(**inputs).logits
        logits = torch.nn.functional.interpolate(logits, size=image_rgb.shape[:2], mode="bilinear",
                                                 align_corners=False)
        return logits.argmax(dim=1)[0].cpu().numpy().astype(np.int64)

    def __call__(self, image_rgb: np.ndarray) -> np.ndarray:
        return self.lut[self.ade_labels(image_rgb)]


# ---------------------------------------------------------------- המחשה

def overlay(image_rgb: np.ndarray, report: dict):
    """אדום = רכיב מוזה (עם שם המחלקה), צהוב מקווקו = אלמנט מה-Layout שלא צויר."""
    from PIL import Image, ImageDraw
    img = Image.fromarray(image_rgb).convert("RGB")
    draw = ImageDraw.Draw(img)
    for c in report["hallucinated"]:
        draw.rectangle(c["bbox"], outline=(230, 20, 20), width=3)
        draw.text((c["bbox"][0] + 3, c["bbox"][1] + 2), c["class"], fill=(230, 20, 20))
    for m in report["missed"]:
        x0, y0, x1, y1 = m["bbox"]
        for x in range(x0, x1, 8):
            draw.line([(x, y0), (min(x + 4, x1), y0)], fill=(240, 200, 0), width=2)
            draw.line([(x, y1), (min(x + 4, x1), y1)], fill=(240, 200, 0), width=2)
        for y in range(y0, y1, 8):
            draw.line([(x0, y), (x0, min(y + 4, y1))], fill=(240, 200, 0), width=2)
            draw.line([(x1, y), (x1, min(y + 4, y1))], fill=(240, 200, 0), width=2)
    draw.rectangle([0, 0, img.width, 18], fill=(0, 0, 0))
    draw.text((4, 3), f"HOR {report['HOR']:.2f}  recall {report['object_recall']:.2f}  HAR {report['HAR']:.2f}  "
                      f"foreign {report['foreign_components']}", fill=(255, 255, 255))
    return img
