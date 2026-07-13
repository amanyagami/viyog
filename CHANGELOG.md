# Changelog

All notable changes to `viyog` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [0.1.2] — 2026-07-13

### Added
- Ship `py.typed` (PEP 561) so downstream type-checkers see the package's type hints.
- Extra project URLs (Documentation, Demo) on the PyPI page; README badges.
- `.github/workflows/test.yml` — lint + pytest on Python 3.10/3.11/3.12.

### Fixed
- Documentation build (`docs/`): corrected the autodoc target to the public
  `viyog` package, switched Napoleon to Google-style, and mock heavy imports so
  Read the Docs builds without installing torch.

## [0.1.1] — 2026-07-13

### Fixed
- `viyog_metrics` returned `NaN` for `AUTC` (and its `AUFPR`/`AUFNR` components)
  because newer scikit-learn prepends an infinite ROC threshold to `roc_curve`,
  which poisoned the trapezoidal integral. Non-finite thresholds are now dropped.

### Added
- `examples/quickstart.py` — runnable ID/OOD/ADV separation demo (real model + PGD
  attack + `viyog_metrics`), with a `--smoke` synthetic mode for offline/CI runs.

## [0.1.0] — 2026-07-13

First functional release: the packaged detector now implements the method from
the paper *"Viyog: Separating Adversarial and Out-of-Distribution."*

### Added
- `Viyog` — training-free, gradient-free detector that scores the **dormant-band
  roughness** `V(x)` of a model's first convolutional layer (magnitude-normalised
  total variation over the quietest in-distribution channels). Higher `V(x)`
  ⇒ more likely adversarial.
  - `fit(id_loader)` learns the dormant band (dead channels excluded) and the ID
    score mean in a single pass.
  - `score(x, center=False)` scores a batch or a whole loader.
  - Supports `Conv2d` (2-D total variation) and `Conv1d` (1-D, for signal models),
    an explicit `layer=` override, `dorm_pct` / `dead_thresh` knobs, context-manager
    hook cleanup, and `bounded_score()` for an optional monotone squash.
- `viyog_metrics(neg, adv)` — OOD/ADV separability report (AUROC, AUPR, FPR@95,
  DetectionError, AUTC). Requires the optional `metrics` extra.

### Changed
- **Replaced** the pre-release `Viyog` prototype (a first-layer L∞ score, which is
  near-chance at OOD-vs-ADV) with the paper's dormant-band roughness method. The
  packaged detector now reproduces the paper's per-model AUROCs exactly.
- Slimmed runtime dependencies to `torch` + `numpy`; `scikit-learn` is now an
  optional extra (`pip install "viyog[metrics]"`).

[0.1.0]: https://pypi.org/project/viyog/0.1.0/
