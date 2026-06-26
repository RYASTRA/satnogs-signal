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

📄 **Background research:** [docs/prior-art.md](docs/prior-art.md) — a cited survey of
prior efforts, how SatNOGS vets today, the labeling trap, and what it means for scope.
