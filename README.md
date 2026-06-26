# satnogs-signal

**Observation vetting / signal detection at scale.**

The SatNOGS network generates a firehose of observations, and a large fraction are
empty — no signal, just noise. Deciding *"did this observation actually catch the
bird?"* still comes down to a human looking at the waterfall image. The network's
*automated* rating only checks whether data was decoded — a heuristic the maintainers
themselves removed once, because it happily marked noise as "good." Waterfall vetting
itself remains entirely manual.

A model that looks at the waterfall image and classifies **signal-vs-noise** — even
just to triage — would improve data quality across the whole network. It can live as a
standalone service that consumes the SatNOGS API and posts results back.

This is the one I'd bet on:

- **Clean CV/ML problem.** A real signal shows up as a roughly vertical, Doppler-curved
  line against background noise — exactly what a CNN is good at.
- **Labels already exist — with a catch.** Train on the *manual* "With Signal / Without
  Signal" waterfall vetting, **not** the automated Good/Bad observation status (which
  would just relearn the broken decode heuristic).
- **Maintainer-endorsed and genuinely open.** There is no deployed waterfall classifier
  in SatNOGS; the team explicitly named ML-on-waterfalls as the intended replacement for
  the auto-vetting they pulled. The only existing attempt is an independent CNN
  proof-of-concept — a baseline to beat, not a solved problem.

## Results (v1)

A compact **ResNet-18** fine-tuned on gold human waterfall vettings, evaluated on a
**leakage-safe held-out test set** — *unseen ground stations **and** an entirely unseen
satellite* (436 observations):

| Metric | **Model** | Classical baseline |
|--------|-----------|--------------------|
| ROC-AUC | **0.801** | 0.557 |
| PR-AUC | **0.800** | 0.558 |
| precision@10 | **1.000** | 0.700 |

- **Generalizes to an unseen satellite** — FrontierSat, never seen in training (240 obs): ROC-AUC **0.760**.
- **precision@10 = 1.000** — the model's most-confident signal calls are all correct, so the
  *top of the triage queue is trustworthy* (the point: this is a triage aid, not an auto-vetter).
- Beats the label-free classical baseline (central-frequency energy) by a wide margin — the v1 ship bar.

**Artifacts** on the Hugging Face Hub:
- 🤖 Model — [`ryroeu/satnogs-signal-classifier`](https://huggingface.co/ryroeu/satnogs-signal-classifier)
- 📊 Dataset — [`ryroeu/satnogs-signal-waterfalls`](https://huggingface.co/datasets/ryroeu/satnogs-signal-waterfalls)
- 📈 Full eval (per-mode / per-satellite slices) — [docs/eval-report.md](docs/eval-report.md)

*Trained on 4 narrowband FSK/GFSK telemetry satellites (OTP-2, CUBEBEL-2, AEPEX, CatSat) with
FrontierSat held out. Caveat: gold labels skew toward clearer passes than the unvetted firehose,
so real-world performance on faint signals will be lower than these numbers.*

📄 **Background research:** [docs/prior-art.md](docs/prior-art.md) — a cited survey of
prior efforts, how SatNOGS vets today, the labeling trap, and what it means for scope.
