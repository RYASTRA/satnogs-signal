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

## The SatNOGS fleet

Four small, honest tools around a SatNOGS observation — three single-purpose engines, plus one app
that composes them for a human reviewer:

| repo | the question it answers |
|---|---|
| **satnogs-signal** (this repo) | ***is there a signal in this waterfall?*** — signal-vs-noise triage |
| [satnogs-decoder](https://github.com/RYSATNOGS/satnogs-decoder) | *what does the frame say?* — telemetry decoding |
| [satnogs-id](https://github.com/RYSATNOGS/satnogs-id) | *which catalog object is it?* — Doppler identification |
| [satnogs-dashboard](https://github.com/RYSATNOGS/satnogs-dashboard) | *review it all on one observation* — the workbench that runs the three engines |

The three engines are standalone and read-only against SatNOGS; the dashboard is the surface that
composes them. This repo is the **detect** stage — a waterfall in, a signal/noise call out.

## Results (v1)

A compact **ResNet-18** fine-tuned on gold human waterfall vettings, evaluated on a
**held-out test set** (436 observations) that combines two axes held out of training: an
**entirely unseen satellite** (FrontierSat, never trained on) and a set of **held-out ground
stations** whose noise/RFI fingerprint is kept out of training:

| Metric | **Model** | Classical baseline |
|--------|-----------|--------------------|
| ROC-AUC | **0.827** | 0.570 |
| PR-AUC | **0.829** | 0.557 |
| precision@10 | **1.000** | 0.600 |

- **Generalizes to an unseen satellite** — FrontierSat, never seen in training (240 obs): ROC-AUC **0.772**.
- **precision@10 = 1.000** *on the gold test set* — on held-out **vetted** passes the
  most-confident calls are all correct. On the raw unvetted firehose the top of the queue is a
  strong prioritizer but still needs a human glance (see *Expected accuracy on live data* below) —
  this is a triage aid, not an auto-vetter.
- Beats the label-free classical baseline (central-frequency energy) by a wide margin — the ship bar.
- Inputs are cropped to the spectrogram (colorbar/axes removed), which re-centers the signal — by
  mode: GFSK **0.93**, FSK **0.92**, FSK AX.100 Mode 5 **0.79**.

**Artifacts** on the Hugging Face Hub:
- 🤖 Model — [`ryroeu/satnogs-signal-classifier`](https://huggingface.co/ryroeu/satnogs-signal-classifier)
- 📊 Dataset — [`ryroeu/satnogs-signal-waterfalls`](https://huggingface.co/datasets/ryroeu/satnogs-signal-waterfalls)

*Trained on 4 narrowband FSK/GFSK telemetry satellites (OTP-2, CUBEBEL-2, AEPEX, CatSat) with
FrontierSat held out. Two caveats: (1) the unseen-satellite slice is held out by satellite, so
some of its passes come from stations that also appear in training — that slice measures
cross-satellite generalization, not fully station-unseen performance; (2) gold labels skew toward
clearer passes than the unvetted firehose, so real-world performance on faint signals will be
lower than these numbers.*

### Expected accuracy on live data

The numbers above are measured on *human-vetted* passes. To check behavior on the **raw
unvetted firehose**, the triage service was run live against the SatNOGS Network API and scored
**56 fresh observations** it had never seen. Independent reviewers then classified the queue's
extremes — the 6 highest- and 6 lowest-scored waterfalls — **blind** (without seeing the model's
score):

| Queue segment | Model call | Blind reviewers agreed |
|---------------|------------|------------------------|
| **Bottom** — P(signal) < 0.02 | "noise" | **6 / 6** — all confirmed noise |
| **Top** — P(signal) ≥ 0.995 | "signal" | **4 / 6** — two confident false positives |

In practice, then: the model is **excellent at filtering out empty observations** (the bulk of
the firehose), while the **top of the queue still needs a human glance** — two of its six
most-confident signal calls were false positives on this live sample. Use it to *prioritize*
vetting, not to auto-accept.

*This is a 12-observation blind spot-check of the queue extremes (one reviewer per image), not a
full metric — it illustrates the real-world behavior the sampling-bias caveat predicts, on fresh
observations the model had not previously scored.*

## Running it (Docker — no virtualenv)

Everything runs in a container; there's no local Python environment to manage. The model is
pre-trained and loaded from the Hub
([`ryroeu/satnogs-signal-classifier`](https://huggingface.co/ryroeu/satnogs-signal-classifier)).

```bash
docker compose build   # build the image (installs runtime deps)
```

API tokens go in a gitignored `.env` (copy `.env.example`); compose loads it automatically.

### The triage service (read-only)

```bash
docker compose run --rm app python scripts/run_poller.py        # score unvetted obs -> triage.db
docker compose run --rm --service-ports app python app.py       # serve the dashboard at http://localhost:7860
```

The poller pulls *unvetted* observations for the model's satellites, scores them, and stores
the ranked predictions; the dashboard shows a **triage queue** (most-likely-signal first, with a
link to vet each on SatNOGS) and a **monitoring** view. It never writes back to SatNOGS.
