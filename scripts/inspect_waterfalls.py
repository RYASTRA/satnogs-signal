"""Render a contact sheet of sample waterfalls per class, to decide the crop box.
Usage: python scripts/inspect_waterfalls.py _dataset_build [n_per_class]"""
import sys

from PIL import Image

from satnogs_signal.model.data import load_splits

source = sys.argv[1] if len(sys.argv) > 1 else "_dataset_build"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 16
dd = load_splits(source)
train = dd["train"]
for label, name in [(1, "with-signal"), (0, "without-signal")]:
    imgs = [r["image"] for r in train if r["label"] == label][:n]
    if not imgs:
        continue
    w, h = imgs[0].size
    cols = 4
    rows = (len(imgs) + cols - 1) // cols
    sheet = Image.new("RGB", (w * cols, h * rows), (0, 0, 0))
    for i, im in enumerate(imgs):
        sheet.paste(im.convert("RGB"), ((i % cols) * w, (i // cols) * h))
    out = f"_dataset_build/sheet_{name}.png"
    sheet.save(out)
    print(f"wrote {out} ({len(imgs)} images)")
