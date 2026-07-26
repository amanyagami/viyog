---
title: Viyog OOD-vs-ADV Leaderboard
emoji: 🛡️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: mit
short_description: Separating adversarial from OOD inputs at the first conv layer
---

# 🛡️ Viyog vs `pytorch-ood` — OOD-vs-ADV Leaderboard

Interactive comparison of **Viyog** (a training-free, gradient-free, sub-KB
first-conv detector) against standard [`pytorch-ood`](https://pytorch-ood.readthedocs.io)
baselines on the task of telling **adversarial (ADV)** inputs apart from
**out-of-distribution (OOD)** inputs — across 20 architectures and 3 datasets.

| Tab | What it shows |
|---|---|
| 🏆 **Leaderboard** | Detectors ranked by T3 (OOD-vs-ADV AUROC), with T1/T2 and memory cost. |
| ⚖️ **Cost vs accuracy** | Viyog sits top-left: highest T3 at sub-KB state; distance detectors need 7–25 MB. |
| 🧩 **Per-architecture** | Per-model T3, Viyog-D vs any baseline you pick. |
| 🎯 **OOD difficulty** | T3 split by far / near / texture OOD. |
| ℹ️ **About** | What T1/T2/T3 mean and how Viyog works. |

## Run locally

```bash
pip install -r requirements.txt   # or: uv run --with-requirements requirements.txt python app.py
python app.py                     # opens http://127.0.0.1:7860
```

## Refresh the data

The app serves the small CSVs in `data/` (≈38 KB). Regenerate them from the
experiment results with:

```bash
VIYOG_ROOT=/path/to/viyog python export_leaderboard.py
```

Source data: `results/analysis/*.csv` — Viyog, CODES+ISSS 2026 (paper #215).
