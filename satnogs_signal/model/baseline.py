"""Label-free classical baseline: the floor the trained model must beat.

The SatNOGS wiki defines a signal as a (roughly) vertical line near the center
of the waterfall. This scores the ratio of central-column brightness to the rest
— crude on purpose, so beating it is meaningful."""
from __future__ import annotations

import numpy as np
from PIL import Image


def signal_score(img: Image.Image, center_frac: float = 0.25) -> float:
    a = np.asarray(img.convert("L"), dtype=float)
    w = a.shape[1]
    c0 = int(w * (0.5 - center_frac / 2))
    c1 = int(w * (0.5 + center_frac / 2))
    center = a[:, c0:c1].mean()
    rest_pixels = np.concatenate([a[:, :c0].ravel(), a[:, c1:].ravel()])
    rest = rest_pixels.mean() + 1e-6
    return float(center / rest)


def score_images(images, center_frac: float = 0.25) -> list:
    return [signal_score(im, center_frac) for im in images]
