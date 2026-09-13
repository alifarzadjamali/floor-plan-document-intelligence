from __future__ import annotations

from transformers import SegformerForSemanticSegmentation

from floorplan_di.constants import CLASS_NAMES


def build_segformer(model_name: str) -> SegformerForSemanticSegmentation:
    id2label = {index: name for index, name in enumerate(CLASS_NAMES)}
    label2id = {name: index for index, name in id2label.items()}
    return SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=len(CLASS_NAMES),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
