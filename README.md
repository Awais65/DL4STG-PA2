# DL4STG PA2: AI651 Assignment 2 (Fall 2026)

- **Q1 (River Forecasting Proposal):** a written design; it lives in the report.
- **Q2 (Motion or Memory?):** ConvLSTM forecasters on Moving MNIST, code in `q2/`.

## Setup

```bash
conda activate ml310
pip install -r requirements.txt
```

Run everything from the repo root as modules, e.g. `python -m q2.checks`.
Moving MNIST is downloaded to `data/` (git-ignored) on first use.

## Layout

| Path | Contents |
|------|----------|
| `q2/data.py` | loading and the manual video split, mirror/rotation transforms, the symmetry group G, speed windows |
| `q2/checks.py` | Table 1 sanity checks |
| `q2/flow.py` | Figure 1: Farneback optical-flow statistics on the training set |
| `q2/models.py` | ConvLSTM forecaster (model A; configurable for model D) |
| `predictions/` | Q2.5 and Q2.11 predictions, each committed **before** its experiments |
| `results/` | tables and per-video metric arrays |
| `figures/`, `gifs/` | report figures and the animated GIFs |
| `AI_USAGE_LOG.md` | prompt / output / edits log for the report's AI appendix |
