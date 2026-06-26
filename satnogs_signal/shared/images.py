"""Waterfall image loading + preprocessing (v1: convert-RGB + resize; crop deferred)."""
from __future__ import annotations

import io

from PIL import Image

INPUT_SIZE = (224, 224)

# Fractional crop of the spectrogram region, decided by inspecting real waterfalls
# (scripts/inspect_waterfalls.py). Identity crop = all zeros.
# Waterfalls have a colorbar/axes border, but since it is constant across both classes
# we defer cropping; this hook returns the image unchanged until a non-identity crop
# is warranted.
_CROP = {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0}


def load_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


def crop_waterfall(img: Image.Image) -> Image.Image:
    w, h = img.size
    box = (
        int(w * _CROP["left"]),
        int(h * _CROP["top"]),
        w - int(w * _CROP["right"]),
        h - int(h * _CROP["bottom"]),
    )
    return img.crop(box) if box != (0, 0, w, h) else img


def preprocess(img: Image.Image, size=INPUT_SIZE) -> Image.Image:
    return crop_waterfall(img).convert("RGB").resize(size, Image.Resampling.LANCZOS)
