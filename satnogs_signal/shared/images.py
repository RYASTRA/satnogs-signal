"""Waterfall image loading + preprocessing (convert-RGB + crop-to-spectrogram + resize)."""

from __future__ import annotations

import io

from PIL import Image

INPUT_SIZE = (224, 224)

# Fractional crop to the spectrogram (plot) region, measured from a full-res SatNOGS
# waterfall (823x1603): a left Y-axis (~13%), bottom X-axis (~6%), small top margin
# (~3%), and a right colorbar + its labels (~22%). Cropping these off removes constant
# clutter AND re-centers the signal (which sits at the plot-area center, ~0.45 of the
# full width because of the left axis + right colorbar).
_CROP = {"left": 0.13, "right": 0.22, "top": 0.03, "bottom": 0.06}


def load_image(data: bytes) -> Image.Image:
    """Decode raw image bytes into a PIL image (lazy; decode errors surface on use)."""
    return Image.open(io.BytesIO(data))


def crop_waterfall(img: Image.Image) -> Image.Image:
    """Crop off the SatNOGS axes/colorbar margins, re-centering the spectrogram."""
    w, h = img.size
    box = (
        int(w * _CROP["left"]),
        int(h * _CROP["top"]),
        w - int(w * _CROP["right"]),
        h - int(h * _CROP["bottom"]),
    )
    return img.crop(box) if box != (0, 0, w, h) else img


def preprocess(img: Image.Image, size=INPUT_SIZE) -> Image.Image:
    """Crop, convert to RGB, and resize a waterfall to the model input size."""
    return crop_waterfall(img).convert("RGB").resize(size, Image.Resampling.LANCZOS)
