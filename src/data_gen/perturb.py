"""Adversarial / distribution-shift perturbations for the eval harness.

These are NOT tampering — they simulate the "noisy, real-world input" conditions
the JD explicitly calls out ("evaluation harnesses against noisy, adversarial, and
real-world inputs"): phone photos, re-scans, compression, skew. A model that only
does well on clean, centered scans is not useful for a real identity-verification
pipeline, so we score every model on this set separately from the clean test set.
"""
import io
import random

import numpy as np
from PIL import Image, ImageFilter


def jpeg_recompress(image: Image.Image, quality=None) -> Image.Image:
    quality = quality or random.randint(20, 45)
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def gaussian_noise(image: Image.Image, sigma=None) -> Image.Image:
    sigma = sigma or random.uniform(8, 20)
    arr = np.array(image.convert("RGB")).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def gaussian_blur(image: Image.Image, radius=None) -> Image.Image:
    radius = radius or random.uniform(1.0, 2.5)
    return image.filter(ImageFilter.GaussianBlur(radius))


def rotate_skew(image: Image.Image, angle=None) -> Image.Image:
    angle = angle if angle is not None else random.uniform(-8, 8)
    return image.rotate(angle, expand=True, fillcolor=(255, 255, 255))


def downscale_upscale(image: Image.Image, factor=None) -> Image.Image:
    factor = factor or random.uniform(0.3, 0.5)
    w, h = image.size
    small = image.resize((max(1, int(w * factor)), max(1, int(h * factor))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


PERTURBATIONS = {
    "jpeg_recompress": jpeg_recompress,
    "gaussian_noise": gaussian_noise,
    "gaussian_blur": gaussian_blur,
    "rotate_skew": rotate_skew,
    "downscale_upscale": downscale_upscale,
}


def apply_random_perturbation(image: Image.Image, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    name = random.choice(list(PERTURBATIONS.keys()))
    return PERTURBATIONS[name](image), name


def apply_stacked_perturbations(image: Image.Image, n=2, seed=None):
    """Compose n perturbations for a 'hard adversarial' tier."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    names = random.sample(list(PERTURBATIONS.keys()), k=min(n, len(PERTURBATIONS)))
    out = image
    for name in names:
        out = PERTURBATIONS[name](out)
    return out, names
