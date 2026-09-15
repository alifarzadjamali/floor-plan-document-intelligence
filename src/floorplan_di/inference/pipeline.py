from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from floorplan_di.data.pdf import load_plan_page
from floorplan_di.data.segmentation import letterbox_image_and_mask, normalise_image
from floorplan_di.geometry import link_doors_to_rooms, link_text_to_rooms, vectorize_mask
from floorplan_di.inference.schema import structured_document
from floorplan_di.models.segformer import build_segformer
from floorplan_di.ocr.entities import classify_entity
from floorplan_di.ocr.tesseract_engine import configure_tesseract, recognise_page


class FloorPlanPipeline:
    def __init__(
        self,
        checkpoint: Path,
        segmentation_config: Path,
        ocr_config: Path,
        device: str | None = None,
    ) -> None:
        self.segmentation_config = yaml.safe_load(segmentation_config.read_text(encoding="utf-8"))
        self.ocr_config = yaml.safe_load(ocr_config.read_text(encoding="utf-8"))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = build_segformer(self.segmentation_config["model_name"])
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.to(self.device).eval()
        self.image_size = int(self.segmentation_config["image_size"])
        self.ocr_method = json.loads(
            Path("results/ocr/preprocessing_selection.json").read_text(encoding="utf-8")
        )["best_method"]
        configure_tesseract()

    def segment(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        blank = np.zeros(image.shape[:2], dtype=np.uint8)
        boxed, _ = letterbox_image_and_mask(image, blank, self.image_size)
        tensor = normalise_image(boxed).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(pixel_values=tensor).logits
            logits = torch.nn.functional.interpolate(
                logits,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            probability = torch.softmax(logits[0], dim=0).cpu().numpy()
        height, width = image.shape[:2]
        scale = min(self.image_size / width, self.image_size / height)
        target_width, target_height = round(width * scale), round(height * scale)
        top, left = (self.image_size - target_height) // 2, (self.image_size - target_width) // 2
        cropped = probability[:, top : top + target_height, left : left + target_width]
        restored = np.stack(
            [
                cv2.resize(channel, (width, height), interpolation=cv2.INTER_LINEAR)
                for channel in cropped
            ]
        )
        return restored.argmax(axis=0).astype(np.uint8), restored

    def run(
        self, input_path: Path, page_number: int = 1, pdf_dpi: int = 200
    ) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
        image = load_plan_page(input_path, page_number=page_number, dpi=pdf_dpi)
        mask, probability = self.segment(image)
        geometry = vectorize_mask(mask, probability, **self.ocr_config.get("geometry", {}))
        ocr = recognise_page(
            image, self.ocr_method, tuple(self.ocr_config["tesseract"]["orientation_candidates"])
        )
        entities = [
            {
                "id": f"T{index + 1:02d}",
                "text": token.text,
                "bbox": list(token.bbox),
                "confidence": token.confidence,
                "orientation_degrees": ocr.orientation,
                "entity_type": classify_entity(token.text),
            }
            for index, token in enumerate(ocr.tokens)
        ]
        text_links = link_text_to_rooms(
            entities, geometry["rooms"], self.ocr_config.get("text_room_nearest_threshold", 30.0)
        )
        door_links, graph = link_doors_to_rooms(
            geometry["doors"], geometry["rooms"], self.ocr_config.get("door_room_threshold", 45.0)
        )
        return (
            structured_document(
                input_path,
                image.shape[:2],
                geometry,
                entities,
                text_links,
                door_links,
                graph,
                page_number=page_number,
            ),
            mask,
            image,
        )
