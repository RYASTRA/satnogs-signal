"""Waterfall image loading + preprocessing (v1: convert-RGB + resize; crop deferred)."""
from __future__ import annotations

import io

from PIL import Image

INPUT_SIZE = (224, 224)


def load_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


def preprocess(img: Image.Image, size=INPUT_SIZE) -> Image.Image:
    return img.convert("RGB").resize(size)
