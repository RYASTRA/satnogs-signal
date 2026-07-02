"""Train the classifier, evaluate on the held-out test split vs the baseline,
write the eval report, and optionally push the model + card to the Hub.

Setup:  set -a; source .env; set +a   (HF_TOKEN needed for --push)
Usage:  python scripts/train_and_eval.py [dataset_source] [--push]"""

import sys
from pathlib import Path

from satnogs_signal.model.data import class_weights, load_splits
from satnogs_signal.model.evaluate import evaluate_split
from satnogs_signal.model.train import TrainConfig, predict_scores, train_classifier

MODEL_REPO = "ryroeu/satnogs-signal-classifier"
OUT_DIR = "_model"

args = [a for a in sys.argv[1:] if not a.startswith("--")]
source = args[0] if args else "_dataset_build"
do_push = "--push" in sys.argv

dd = load_splits(source)
weights = class_weights(dd["train"])
print(f"class weights (0,1): {weights}")

MODEL_DIR = train_classifier(
    dd["train"], dd["val"], OUT_DIR, TrainConfig(epochs=5, weights=weights)
)
test = dd["test"]
scores = predict_scores(MODEL_DIR, list(test["image"]))
rep = evaluate_split(test, scores)

lines = [
    "# Eval report\n",
    f"Test rows: {test.num_rows}\n",
    "## Model vs baseline (held-out test)\n",
]
for k in ("roc_auc", "pr_auc", "precision_at_10"):
    lines.append(
        f"- {k}: model={rep['model'][k]:.3f}  baseline={rep['baseline'][k]:.3f}\n"
    )
lines.append("\n## Sliced model ROC-AUC\n")
for key, groups in rep["sliced_by"].items():
    lines.append(f"### by {key}\n")
    for g, m in sorted(groups.items(), key=lambda kv: str(kv[0])):
        auc = "n/a" if m["roc_auc"] is None else f"{m['roc_auc']:.3f}"
        lines.append(f"- {g}: n={m['n']} roc_auc={auc}\n")
Path("docs").mkdir(exist_ok=True)
with open("docs/eval-report.md", "w", encoding="utf-8") as report_file:
    report_file.writelines(lines)
print("wrote docs/eval-report.md")
print(
    f"MODEL roc_auc={rep['model']['roc_auc']:.3f}  "
    f"BASELINE roc_auc={rep['baseline']['roc_auc']:.3f}"
)

if do_push:
    from huggingface_hub import ModelCard
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    AutoModelForImageClassification.from_pretrained(MODEL_DIR).push_to_hub(
        MODEL_REPO, private=True
    )
    AutoImageProcessor.from_pretrained(MODEL_DIR).push_to_hub(MODEL_REPO, private=True)
    card = ModelCard(
        f"# satnogs-signal-classifier\n\n"
        f"ResNet-18 fine-tuned for SatNOGS waterfall signal-vs-noise (narrowband FSK/GFSK "
        f"cubesat telemetry). Trained on gold `waterfall_status` labels.\n\n"
        f"## Held-out test metrics\n"
        f"- model ROC-AUC: {rep['model']['roc_auc']:.3f} "
        f"(baseline {rep['baseline']['roc_auc']:.3f})\n"
        f"- model PR-AUC: {rep['model']['pr_auc']:.3f}\n\n"
        f"## Limits\n"
        f"Sampling bias: gold labels skew toward clearer passes than the unvetted firehose; "
        f"real-world performance on marginal observations will be lower. Trained on a narrow "
        f"satellite/mode family; generalization beyond it is unverified. Read-only triage aid, "
        f"not an auto-vetter."
    )
    card.push_to_hub(MODEL_REPO)
    print(f"pushed model + card to {MODEL_REPO}")
