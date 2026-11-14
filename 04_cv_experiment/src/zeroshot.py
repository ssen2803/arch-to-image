"""Baseline zero-shot: סגמנטציה סמנטית בלי שום אימון על הנתונים שלנו.

SAM לבדו מחזיר מסכות בלי מחלקות. כדי לקבל סגמנטציה סמנטית צריך עיגון טקסטואלי:
  backend "grounded_sam" (ברירת מחדל, פתוח):
      GroundingDINO (טקסט -> תיבות) + SAM (תיבה -> מסכה). זו גישת Grounded-SAM.
  backend "sam3" (Meta SAM 3, "segment anything with concepts"):
      מקבל ביטוי טקסט ומחזיר ישירות מסכות לכל מופע. הוא הכי קרוב ל"SAM עם מחלקות",
      אבל המשקולות ב-Hugging Face נעולות: צריך לבקש גישה ל-facebook/sam3 בחשבון
      שלך ולהריץ huggingface-cli login. הענף הזה כתוב לפי התיעוד של transformers
      ולא נבדק כאן, כי אין גישה.

הרכבת מפת המחלקות: מציירים לפי סדר האונטולוגיה (רצפה, קיר, מתקן, בור), כך שבור
שנחפר ברצפה גובר עליה; בתוך מחלקה, מסכה עם ציון גבוה יותר מצוירת אחרונה.

"כיוונון" ה-baseline (רשימת ביטויים וספים) מותר רק על real-dev, ונקבע לפני הרצת ה-test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

CLASS_ORDER = ["floor", "wall", "installation", "pit"]   # סדר ציור
LABEL = {"background": 0, "floor": 1, "wall": 2, "installation": 3, "pit": 4}
DEFAULT_PROMPTS = {
    "wall": ["stone wall", "mudbrick wall", "wall foundation"],
    "floor": ["plastered floor", "paved floor", "mosaic floor", "packed earth floor"],
    "installation": ["clay oven", "stone hearth", "stone bench", "stone basin"],
    "pit": ["pit", "hole in the ground", "cistern opening"],
}


def _class_of(text_label: str, prompts: dict) -> str | None:
    """GroundingDINO מחזיר את הביטוי שזוהה (לפעמים מאוחד). בוחרים את המחלקה עם הכי הרבה מילים משותפות."""
    words = set(text_label.lower().split())
    best, score = None, 0
    for cls, phrases in prompts.items():
        s = max(len(words & set(p.split())) for p in phrases)
        if s > score:
            best, score = cls, s
    return best


def _compose(shape: tuple[int, int], detections: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=np.uint8)
    for cls in CLASS_ORDER:
        for d in sorted((d for d in detections if d["class"] == cls), key=lambda d: d["score"]):
            out[d["mask"]] = LABEL[cls]
    return out


class GroundedSAM:
    name = "grounded_sam"

    def __init__(self, prompts: dict | None = None, box_threshold: float = 0.3, text_threshold: float = 0.25,
                 max_boxes: int = 40, dino_id: str = "IDEA-Research/grounding-dino-tiny",
                 sam_id: str = "facebook/sam-vit-base", device: str | None = None):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.prompts = prompts or DEFAULT_PROMPTS
        self.box_threshold, self.text_threshold, self.max_boxes = box_threshold, text_threshold, max_boxes
        self.dino_proc = AutoProcessor.from_pretrained(dino_id)
        self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(self.device).eval()
        self.sam_proc = SamProcessor.from_pretrained(sam_id)
        self.sam = SamModel.from_pretrained(sam_id).to(self.device).eval()
        self.config = {"backend": self.name, "prompts": self.prompts, "box_threshold": box_threshold,
                       "text_threshold": text_threshold, "max_boxes": max_boxes, "dino": dino_id, "sam": sam_id}

    @torch.inference_mode()
    def detect(self, image: Image.Image) -> list[dict]:
        text = ". ".join(p for ps in self.prompts.values() for p in ps) + "."
        inputs = self.dino_proc(images=image, text=text, return_tensors="pt").to(self.device)
        outputs = self.dino(**inputs)
        res = self.dino_proc.post_process_grounded_object_detection(
            outputs, inputs["input_ids"], threshold=self.box_threshold, text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]])[0]
        labels = res.get("text_labels", res.get("labels"))
        dets = [{"box": b.tolist(), "score": float(s), "class": _class_of(str(t), self.prompts), "text": str(t)}
                for b, s, t in zip(res["boxes"], res["scores"], labels)]
        dets = sorted((d for d in dets if d["class"]), key=lambda d: -d["score"])[:self.max_boxes]
        if not dets:
            return []
        sam_in = self.sam_proc(images=image, input_boxes=[[d["box"] for d in dets]], return_tensors="pt").to(self.device)
        sam_out = self.sam(**sam_in, multimask_output=False)
        masks = self.sam_proc.post_process_masks(sam_out.pred_masks.cpu(), sam_in["original_sizes"].cpu(),
                                                 sam_in["reshaped_input_sizes"].cpu())[0]
        for d, m in zip(dets, masks):
            d["mask"] = m[0].numpy().astype(bool)
        return dets

    def __call__(self, image: Image.Image) -> np.ndarray:
        return _compose((image.height, image.width), self.detect(image))


class SAM3Concepts:
    """SAM 3 עם הנחיית מושג טקסטואלית. דורש גישה ל-facebook/sam3 (gated). לא נבדק כאן."""
    name = "sam3"

    def __init__(self, prompts: dict | None = None, threshold: float = 0.4, mask_threshold: float = 0.5,
                 model_id: str = "facebook/sam3", device: str | None = None):
        from transformers import Sam3Model, Sam3Processor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.prompts = prompts or DEFAULT_PROMPTS
        self.threshold, self.mask_threshold = threshold, mask_threshold
        self.proc = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id).to(self.device).eval()
        self.config = {"backend": self.name, "prompts": self.prompts, "threshold": threshold,
                       "mask_threshold": mask_threshold, "model": model_id}

    @torch.inference_mode()
    def __call__(self, image: Image.Image) -> np.ndarray:
        dets = []
        for cls, phrases in self.prompts.items():
            for phrase in phrases:
                inputs = self.proc(images=image, text=phrase, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                res = self.proc.post_process_instance_segmentation(
                    outputs, threshold=self.threshold, mask_threshold=self.mask_threshold,
                    target_sizes=[image.size[::-1]])[0]
                for m, s in zip(res["masks"], res["scores"]):
                    dets.append({"class": cls, "score": float(s), "mask": m.cpu().numpy().astype(bool)})
        return _compose((image.height, image.width), dets)


def load_zeroshot(backend: str, prompts_json: Path | None = None, **kwargs):
    prompts = json.loads(prompts_json.read_text(encoding="utf-8")) if prompts_json else None
    if backend == "grounded_sam":
        return GroundedSAM(prompts, **kwargs)
    if backend == "sam3":
        return SAM3Concepts(prompts, **kwargs)
    raise ValueError(f"backend לא מוכר: {backend}")
