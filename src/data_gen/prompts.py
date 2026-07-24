"""Instruction templates for the two fine-tuning tasks. Both share one model
(single LoRA adapter, multi-task instruction tuning) so a single VLM handles
both "read this document" and "is this document real" -- mirroring how a real
identity-verification pipeline needs both capabilities on the same intake image.
"""
import json
import random

OCR_INSTRUCTIONS = [
    "Extract the line items (name, quantity, price) and the total amount from this "
    "receipt as JSON.",
    "Read this receipt and return {items: [{name, qty, price}], subtotal, total} as JSON.",
    "List every item purchased with its price, and the receipt total, as JSON.",
]

FORGERY_INSTRUCTIONS = [
    "Is this document authentic or has it been tampered with? If tampered, briefly "
    "describe what looks altered and roughly where in the image.",
    "Inspect this document image for signs of digital tampering (copy-move, splicing, "
    "or edited text/fields). State authentic or tampered, and explain your reasoning.",
    "Analyze this scanned document for forgery. Report your verdict and, if tampered, "
    "the approximate location and type of tampering.",
]


def ocr_sample(image_id, fields: dict):
    instruction = random.choice(OCR_INSTRUCTIONS)
    answer = json.dumps(
        {
            "vendor": fields.get("vendor", "unknown"),
            "date": fields.get("date", "unknown"),
            "total": fields.get("total", "unknown"),
        },
        ensure_ascii=False,
    )
    return {
        "id": image_id,
        "task": "ocr",
        "instruction": instruction,
        "response": answer,
    }


def _region_desc(image_size, bbox):
    w, h = image_size
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    horiz = "left" if cx < w / 3 else ("right" if cx > 2 * w / 3 else "center")
    vert = "top" if cy < h / 3 else ("bottom" if cy > 2 * h / 3 else "middle")
    if horiz == "center" and vert == "middle":
        return "the center of the document"
    return f"the {vert}-{horiz} region of the document"


TAMPER_TYPE_DESC = {
    "copy_move": "a region appears duplicated elsewhere in the document (copy-move forgery)",
    "splice": "a region appears to have been pasted in from a different source document (splicing)",
    "patch_overlay": "a field looks like it was blanked out and overwritten with different text/values",
}


def forgery_sample(image_id, is_tampered: bool, image_size=None, bbox=None, tamper_type=None):
    instruction = random.choice(FORGERY_INSTRUCTIONS)
    if not is_tampered:
        answer = "Authentic. No signs of copy-move, splicing, or field tampering detected."
    else:
        loc = _region_desc(image_size, bbox) if (image_size and bbox) else "the document"
        desc = TAMPER_TYPE_DESC.get(tamper_type, "the region appears digitally altered")
        answer = f"Tampered. In {loc}, {desc}."
    return {
        "id": image_id,
        "task": "forgery",
        "instruction": instruction,
        "response": answer,
        "label": "tampered" if is_tampered else "authentic",
        "bbox": bbox,
        "tamper_type": tamper_type,
    }


def to_conversation(sample: dict, image_path: str):
    """Qwen2-VL / unsloth chat-format conversation record."""
    return {
        "id": sample["id"],
        "task": sample["task"],
        "image": image_path,
        "meta": {k: v for k, v in sample.items() if k not in ("id", "task", "instruction", "response")},
        "conversations": [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": sample["instruction"]}]},
            {"role": "assistant", "content": [{"type": "text", "text": sample["response"]}]},
        ],
    }
