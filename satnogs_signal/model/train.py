"""Fine-tune a compact ResNet-18 for waterfall signal-vs-noise, class-weighted."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)

_LABELS = ["without-signal", "with-signal"]


def _resolve_size(processor) -> int:
    """Resolve the model's expected square input size from the image-processor config.
    Handles both {"shortest_edge": N} and {"height": H, "width": W} forms."""
    size_info = processor.size
    if "shortest_edge" in size_info:
        return size_info["shortest_edge"]
    if "height" in size_info:
        return size_info["height"]
    return 224


class _WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._w = (
            torch.tensor(class_weights, dtype=torch.float)
            if class_weights is not None
            else None
        )

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        w = self._w.to(outputs.logits.device) if self._w is not None else None
        loss = nn.functional.cross_entropy(outputs.logits, labels, weight=w)
        return (loss, outputs) if return_outputs else loss


def _make_transform(processor, train: bool):
    size = _resolve_size(processor)
    mean, std = processor.image_mean, processor.image_std

    def _t(batch):
        out = []
        for im in batch["image"]:
            im = im.convert("RGB").resize((size, size))
            arr = np.asarray(im, dtype=np.float32) / 255.0
            if train and np.random.rand() < 0.3:  # light brightness jitter
                arr = np.clip(arr * np.random.uniform(0.85, 1.15), 0, 1)
            arr = (arr - mean) / std
            out.append(torch.tensor(arr).permute(2, 0, 1).float())
        batch["pixel_values"] = out
        return batch

    return _t


def train_classifier(
    train_ds,
    val_ds,
    output_dir,
    epochs=3,
    lr=5e-5,
    weights=None,
    model_name="microsoft/resnet-18",
) -> str:
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: _LABELS[0], 1: _LABELS[1]},
        label2id={_LABELS[0]: 0, _LABELS[1]: 1},
        ignore_mismatched_sizes=True,
    )
    train_ds = train_ds.with_transform(_make_transform(processor, train=True))
    val_ds = val_ds.with_transform(_make_transform(processor, train=False))

    def _collate(rows):
        return {
            "pixel_values": torch.stack([r["pixel_values"] for r in rows]),
            "labels": torch.tensor([r["label"] for r in rows]),
        }

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = _WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=_collate,
        class_weights=weights,
    )
    trainer.train()
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    return output_dir


def load_scorer(model_dir):
    """Load the processor + model ONCE and return ``score(images) -> list[float]``
    (P(with-signal) per image). Reuse the returned callable across many batches to
    avoid reloading the model on every call (e.g. in the poller)."""
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir).eval()
    size = _resolve_size(processor)
    mean, std = processor.image_mean, processor.image_std

    def score(images) -> list:
        scores = []
        with torch.no_grad():
            for im in images:
                arr = (
                    np.asarray(im.convert("RGB").resize((size, size)), dtype=np.float32)
                    / 255.0
                )
                arr = (arr - mean) / std
                t = torch.tensor(arr).permute(2, 0, 1).float().unsqueeze(0)
                prob = torch.softmax(model(pixel_values=t).logits, dim=1)[0, 1]
                scores.append(float(prob))
        return scores

    return score


def predict_scores(model_dir, images) -> list:
    """One-shot convenience wrapper: load the model and score ``images``.
    For repeated scoring (the poller), use ``load_scorer`` and reuse its closure."""
    return load_scorer(model_dir)(images)
