"""Synthetic document tampering: generates (tampered_image, bbox, tamper_type) pairs
from genuine document images. Standard approach for forgery-detection datasets when
real forged IDs/receipts aren't available or legal to use (cf. DocTamper, FCD).

Tamper types:
  - copy_move: a region is copied and pasted elsewhere on the same document
  - splice: a region from a *different* document is pasted in (donor splicing)
  - patch_overlay: a region is blanked/recolored and re-rendered with different text,
    simulating a field edit (e.g. altering a total amount)
"""
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class TamperResult:
    image: Image.Image
    bbox: tuple  # (x0, y0, x1, y1) in the tampered image's coordinate space
    tamper_type: str


def _random_region(w, h, min_frac=0.08, max_frac=0.25):
    rw = int(w * random.uniform(min_frac, max_frac))
    rh = int(h * random.uniform(min_frac, max_frac))
    rw, rh = min(rw, w - 1), min(rh, h - 1)
    x0 = random.randint(0, w - rw - 1)
    y0 = random.randint(0, h - rh - 1)
    return x0, y0, x0 + rw, y0 + rh


def _content_region(w, h, content_boxes, pad=6):
    """Pick a region anchored on a real text line (from CORD's word boxes)
    instead of a uniformly random patch of the image, so tampering lands on
    document content rather than blank background."""
    if not content_boxes:
        return _random_region(w, h)
    x0, y0, x1, y1 = random.choice(content_boxes)
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return _random_region(w, h)
    return x0, y0, x1, y1


def copy_move(image: Image.Image, content_boxes=None, seed=None) -> TamperResult:
    if seed is not None:
        random.seed(seed)
    img = image.copy()
    w, h = img.size
    src_box = _content_region(w, h, content_boxes)
    region = img.crop(src_box)

    rw, rh = src_box[2] - src_box[0], src_box[3] - src_box[1]
    dst_candidates = [b for b in (content_boxes or []) if b != src_box]
    x0 = y0 = None
    for _ in range(20):
        if dst_candidates:
            cb = random.choice(dst_candidates)
            x0 = max(0, min(w - rw - 1, cb[0]))
            y0 = max(0, min(h - rh - 1, cb[1]))
        else:
            x0 = random.randint(0, max(0, w - rw - 1))
            y0 = random.randint(0, max(0, h - rh - 1))
        # avoid pasting back onto (near) the same spot
        if abs(x0 - src_box[0]) > rw or abs(y0 - src_box[1]) > rh:
            break
    dst_box = (x0, y0, x0 + rw, y0 + rh)

    if random.random() < 0.5:
        region = region.transpose(Image.FLIP_LEFT_RIGHT)

    img.paste(region, dst_box[:2])
    return TamperResult(img, dst_box, "copy_move")


def splice(image: Image.Image, donor_image: Image.Image, content_boxes=None, seed=None) -> TamperResult:
    if seed is not None:
        random.seed(seed)
    img = image.copy()
    w, h = img.size
    donor = donor_image.resize((w, h))

    src_box = _content_region(w, h, content_boxes)
    region = donor.crop(src_box)
    img.paste(region, src_box[:2])
    return TamperResult(img, src_box, "splice")


def patch_overlay(image: Image.Image, content_boxes=None, seed=None) -> TamperResult:
    """Simulates a field edit: blank a small region and draw different digits/text
    over it, mimicking an altered total/date/ID number."""
    if seed is not None:
        random.seed(seed)
    img = image.copy()
    w, h = img.size
    x0, y0, x1, y1 = _content_region(w, h, content_boxes, pad=2)

    region = img.crop((x0, y0, x1, y1))
    bg = np.array(region.convert("RGB")).reshape(-1, 3)
    fill_color = tuple(int(c) for c in np.median(bg, axis=0))

    draw = ImageDraw.Draw(img)
    draw.rectangle([x0, y0, x1, y1], fill=fill_color)

    fake_text = random.choice(["48.90", "129.00", "07/14", "0091273", "TOTAL"])
    font_size = max(10, (y1 - y0) - 4)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except Exception:
        font = ImageFont.load_default()
    text_color = tuple(255 - c for c in fill_color)
    draw.text((x0 + 2, y0 + 1), fake_text, fill=text_color, font=font)

    return TamperResult(img, (x0, y0, x1, y1), "patch_overlay")


def apply_random_tamper(image: Image.Image, donor_pool=None, content_boxes=None, seed=None) -> TamperResult:
    if seed is not None:
        random.seed(seed)
    choices = ["copy_move", "patch_overlay"]
    if donor_pool:
        choices.append("splice")
    choice = random.choice(choices)
    if choice == "copy_move":
        return copy_move(image, content_boxes=content_boxes, seed=seed)
    if choice == "patch_overlay":
        return patch_overlay(image, content_boxes=content_boxes, seed=seed)
    donor = random.choice(donor_pool)
    return splice(image, donor, content_boxes=content_boxes, seed=seed)
