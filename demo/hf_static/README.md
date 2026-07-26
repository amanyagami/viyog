---
title: Viyog OOD-vs-ADV Leaderboard
emoji: 🛡️
colorFrom: green
colorTo: blue
sdk: static
pinned: false
license: mit
short_description: Separating adversarial from out-of-distribution inputs
---

# 🛡️ Viyog vs `pytorch-ood` — OOD-vs-ADV Leaderboard

A self-contained **static** page (no server) comparing **Viyog** — a training-free,
gradient-free, sub-KB first-conv detector — against standard
[`pytorch-ood`](https://pytorch-ood.readthedocs.io) baselines on telling
**adversarial (ADV)** inputs apart from **out-of-distribution (OOD)** inputs,
across 20 architectures and 3 datasets. Charts render client-side with Plotly.

- **Package:** `pip install viyog` — https://pypi.org/project/viyog/
- **Code:** https://github.com/amanyagami/viyog
- Data: `results/analysis/*.csv` (Viyog, CODES+ISSS 2026, paper #215)

Rebuild `index.html` from the source CSVs with `python ../build_static.py`.
