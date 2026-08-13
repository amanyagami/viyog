# Viyog

[![PyPI](https://img.shields.io/pypi/v/viyog.svg)](https://pypi.org/project/viyog/)
[![Python](https://img.shields.io/pypi/pyversions/viyog.svg)](https://pypi.org/project/viyog/)
[![Downloads](https://static.pepy.tech/badge/viyog)](https://pepy.tech/project/viyog)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/amanyagami/viyog/blob/main/LICENSE)
[![Live demo](https://img.shields.io/badge/%F0%9F%A4%97-Live%20demo-009E73)](https://huggingface.co/spaces/amanyagami/viyog)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21918573-blue)](https://doi.org/10.5281/zenodo.21918573)

**[▶ Try the interactive leaderboard](https://huggingface.co/spaces/amanyagami/viyog)** — Viyog vs 9 `pytorch-ood` baselines across 20 architectures.

**Separate adversarial (ADV) inputs from out-of-distribution (OOD) inputs — in one forward pass, with no training and no gradients.**

Safety-critical systems must respond *differently* to two kinds of anomaly: OOD
inputs call for **abstention**, adversarial inputs demand **rejection**. Standard
detectors collapse both into a single anomaly score and cannot tell them apart.
Viyog is a training-free, post-hoc second stage that makes the distinction by
reading the **dormant-band roughness** of a model's first convolutional layer.

The idea: gradient-based attacks inject broadband high-frequency residue into the
first-layer channels that are otherwise *quiet* on in-distribution data, making
them spatially jagged. Natural inputs — both ID and OOD — leave those channels
smooth. Viyog measures that roughness as a single scalar `V(x)`:

* **higher `V(x)` → more likely ADV**
* **lower `V(x)` → more likely OOD / ID**

It adds no parameters, never touches the backward pass, and stores only `O(C)`
bytes of state (the dormant-channel ranking + one ID mean) — roughly **0.3 KB**,
versus 4.5–40 MB for feature-distance detectors such as Mahalanobis / KNN / ViM.

## Install

```bash
pip install "viyog==0.1.3"          # core: torch + numpy
pip install "viyog[metrics]==0.1.3" # + scikit-learn, for viyog_metrics()
```

## Canonical artifacts

The following are the maintained artifacts for this paper (verified
2026-08-08). Use the immutable revisions when reproducing or citing the work.

| Artifact | Canonical version | Purpose |
| --- | --- | --- |
| [Zenodo archive](https://doi.org/10.5281/zenodo.21918573) | AE v6, DOI `10.5281/zenodo.21918573` | Version-specific archival citation. The [concept DOI](https://doi.org/10.5281/zenodo.21614317) is the collection landing page. |
| [GitHub source](https://github.com/amanyagami/viyog) | [`v0.1.3-codes26-ae-6`](https://github.com/amanyagami/viyog/releases/tag/v0.1.3-codes26-ae-6) | Package, experiments, paper source, and reproduction entry point. |
| [PyPI package](https://pypi.org/project/viyog/0.1.3/) | `viyog==0.1.3` | Installable dormant-band-TV detector. |
| [Model weights](https://huggingface.co/amanyagami/viyog-weights/tree/b96263ae28a8585acaad6a002a0298b6b6c2d735) | revision `b96263a`; 34 checkpoints, 5.00 GB (4.66 GiB) | Fine-tuned backbones. The six-model T1 subset is 1.58 GB (1.47 GiB). |
| [Optional adversarial data](https://huggingface.co/datasets/amanyagami/viyog-adversarial/tree/5e15272b1023fbfb8420242c7da64884aacb9b91) | revision `5e15272`; 40 HDF5 files; 66.09 GB repository storage (41.84 GB current file payload) | Supplemental precomputed attacks. The reproduction script does **not** consume these files; it regenerates attacks. |
| [Optional live Space](https://huggingface.co/spaces/amanyagami/viyog) | [revision `c8d4c9f`](https://huggingface.co/spaces/amanyagami/viyog/tree/c8d4c9fbbbf8f33372bc428d122ba271a202a405), running when verified | Static interactive results explorer, not the reproduction pipeline. |
| Documentation source | [`docs/`](docs/) in this repository | Updated dormant-band API documentation. The public Read the Docs deployment is stale until it is reconnected and rebuilt, so it is not a canonical artifact yet. |

The Zenodo v5 DOI and PyPI 0.1.3 are the intended latest releases. Historical
infinity-norm prototypes, the old `MPSLab-ASU/Seperating_OOD_and_ADV` repository,
the `Cifar100_Finetuned` weights, the empty ViT placeholder, and the old Google
Drive folder are legacy artifacts and are **not** supported reproduction sources.

## Quickstart

```python
from viyog import Viyog

v = Viyog(model)          # attaches a forward hook to the first conv layer
v.fit(id_loader)          # one ID pass: learn the dormant band + ID mean
scores = v.score(loader)  # per-sample roughness V(x); HIGHER => more adversarial
v.close()
```

Or as a context manager (hooks are removed automatically):

```python
from viyog import Viyog, viyog_metrics

with Viyog(model) as v:
    v.fit(id_loader)
    ood_scores = v.score(ood_loader)
    adv_scores = v.score(adv_loader)

# separability report (OOD vs ADV): AUROC / AUPR / FPR@95 / AUTC
print(viyog_metrics(ood_scores.cpu().numpy(), adv_scores.cpu().numpy()))
```

### With a real model (timm)

Viyog auto-detects the first convolutional layer (`resnet50.conv1` here), so
models with a `Conv1d` or `Conv2d` feature map work out of the box:

```python
import torch, timm
from viyog import Viyog

model = timm.create_model("resnet50", pretrained=True, num_classes=10).eval()

with Viyog(model) as v:
    v.fit(id_loader)                 # your in-distribution DataLoader
    print(v.layer_name_, v.n_channels_, v.dorm_idx_.numel())   # conv1 64 6
    adv_scores = v.score(suspect_loader)   # higher V(x) => more adversarial
```

Pass `device="cuda:0"` for GPU, or `layer=<module or dotted name>` to hook a
specific layer instead of the auto-detected first conv.

## How it works

For each input, Viyog hooks the first conv layer's activation map `a` of shape
`(B, C, H, W)` and computes, per channel, a **magnitude-normalised total
variation** (average absolute change between neighbouring pixels):

```
tv = (|Δ_h a| + |Δ_w a|) / (mean|a| + eps)      # spatial roughness, per channel
V(x) = mean over the dormant channels of tv     # the Viyog score
```

The **dormant band** is the quietest `dorm_pct` (default 10%) of channels ranked
by their mean absolute activation on ID data, restricted to *alive* channels
(permanently-dead channels — common in e.g. DenseNet first convs — are excluded so
the statistic stays meaningful). `.fit()` learns that band and the ID mean of `V`
in a single pass; `.score()` returns `V(x)` for new inputs.

`Conv2d` maps use 2-D total variation; `Conv1d` maps `(B, C, L)` use 1-D total
variation, so the detector also applies to 1-D signal models.

## API

| | |
| --- | --- |
| `Viyog(model, device=None, layer=None, dorm_pct=0.10, dead_thresh=1e-4)` | Wrap a model; auto-detects the first conv layer (prefers a `conv1` attribute). Pass `layer=` (module or dotted name) to override. |
| `.fit(id_loader) -> self` | One ID pass: selects the dormant band, records `id_score_mean_`. |
| `.score(x, center=False) -> Tensor` | Per-sample `V(x)` for a batch **or** a whole loader. Higher = more adversarial. `center=True` subtracts the ID mean (monotone; AUROC-preserving). |
| `.score_loader(loader, center=False)` | Convenience wrapper over a loader. |
| `Viyog.bounded_score(scores, temperature=1.0)` | Optional monotone squash to `(0, 1)` for thresholding/display. |
| `.close()` / context manager | Remove the forward hook. |
| `viyog_metrics(neg, adv, recall_level=0.95)` | AUROC / AUPR_IN / AUPR_OUT / FPR95 / DetectionError / AUTC for two score populations. Needs `viyog[metrics]`. |

Fitted attributes: `dorm_idx_`, `id_profile_`, `id_score_mean_`, `n_channels_`,
`layer_name_`.

## Results

Across the 17-architecture held-out panel on CIFAR-100 (ResNet/DenseNet/ConvNeXt/Swin/ViT
and edge backbones — MobileOne-S1, ResNet-18, and ResNet-34 are excluded because their
archived first-layer feature files were found to be corrupted, see STATUS), the
dormant-band roughness score reaches **AUROC ≈ 0.980 for adversarial
detection (ID vs ADV)** and **≈ 0.844 for OOD vs ADV** — where logit detectors
(Energy/MSP/MaxLogit/GEN) are near-blind to adversarials and feature-distance
detectors cost 4.5–40 MB of state versus Viyog's ~0.3 KB. See the accompanying paper.

## Reproducing the paper

This repository is the CODES+ISSS 2026 journal-track artifact for the
accompanying paper, accepted for publication in *IEEE Transactions on
Computer-Aided Design of Integrated Circuits and Systems* (IEEE TCAD). Besides
the installable package (`src/viyog/`), it contains the full reproduction
pipeline (`experiments/`), the paper source (`paper_rev/`), and an interactive
leaderboard (`demo/`). Nothing the core tier reports is shipped as a precomputed
result to replay: the numbers below are regenerated from the public checkpoints
and benchmark datasets. See [REQUIREMENTS](REQUIREMENTS), [INSTALL](INSTALL),
and [STATUS](STATUS) for details.

| Tier | What it does | Hardware | Badge |
| --- | --- | --- | --- |
| **T0 — sanity** | `pytest` + `examples/quickstart.py --smoke` | CPU, ~2 min | Reviewed |
| **T1 — core recompute** | Fetch the 6 core checkpoints, regenerate adversarial examples, extract first-conv features, recompute every signature's AUROC — from scratch, nothing cached | 1 CUDA GPU, 4+ hours (first run; much faster on a re-run) | **Reproducible** |
| **T2 — extended** | Same pipeline, the 17-architecture held-out panel + cifar10 | 1 CUDA GPU, longer | extra evidence |
| **T3 — from scratch** | Re-finetune all 20 backbones from raw ImageNet weights | GPU-weeks | documented, out of scope |

**Reviewers: one command reproduces Tier T1 end to end.**

```bash
bash reproduce_t1.sh --quick    # smoke test — does the pipeline work here?
bash reproduce_t1.sh            # full Tier T1 — reproduce the paper
```

The wrapper downloads the six checkpoints from immutable Hugging Face revision
`b96263ae28a8585acaad6a002a0298b6b6c2d735` (1.58 GB / 1.47 GiB). The optional
41.84 GB adversarial-data snapshot is not downloaded: adversarial examples are
regenerated locally.

`reproduce_t1.sh` preflights your machine (GPU, VRAM, disk), **scales both GPU
batch sizes to your card** (the committed defaults are sized for a 143 GB H200;
profiles defined from 8 GB upward), runs the whole pipeline, verifies that
every expected feature split was produced, and prints the measured T2/T3 AUROC
beside the values claimed below. `--quick` runs the same pipeline on one backbone
/ one attack / two OOD sets so you can confirm the artifact works before
committing hours to it.

Both tiers are dominated by **first-run dataset downloads**, not compute:
`--quick` is ~2 minutes of GPU work but fetches ~350 MB of benchmark data the
first time (30-60 min on a throttled link; minutes once cached). The full tier
measured 4 h 7 m end to end on a clean clone with nothing cached.

The individual steps below remain available if you would rather drive them
yourself (see INSTALL for the full walkthrough):

```bash
uv sync --frozen --group experiments
uv run --frozen python experiments/01_download.py --core-only
uv run --frozen python experiments/03_gen_adversarial.py \
    --models convnextv2_base densenet121 mobilenetv3_l resnet50 swin_tiny vit_base \
    --attacks fgsm bim pgd apgd_ce
uv run --frozen python experiments/06b_extract_full.py \
    --models convnextv2_base densenet121 mobilenetv3_l resnet50 swin_tiny vit_base \
    --attacks fgsm bim pgd apgd_ce
uv run --frozen python experiments/09_signatures_full.py \
    --models convnextv2_base densenet121 mobilenetv3_l resnet50 swin_tiny vit_base
uv run --frozen python experiments/full_eval.py --dataset cifar100
```

Expected headline numbers for **this 6-model core tier** (CIFAR-100,
`ViyogD_tv_dorm` — the deployed detector, `src/viyog`'s own `tv` statistic),
as produced by the exact commands above:

| Metric | Value |
| --- | --- |
| T2 (ID-vs-ADV) AUROC | **~0.98** (measured: 0.9804) |
| T3 (OOD-vs-ADV) AUROC | **~0.84** (measured: 0.8441) |

The paper's two efficiency headlines — **~0.3 KB** of detector state and
**2.284%** of a full forward pass — are *not* produced by the commands above.
They are architecture-only measurements over the paper's panel and need no GPU,
no weights and no dataset, so reproduce them separately in seconds:

```bash
uv run --frozen python experiments/eval_detector_cost.py   # state memory, per model
uv run --frozen python experiments/eval_systems.py         # fvcore MAC ratios
```

State scales linearly with the first conv's output-channel count, so read the
per-model rows rather than only the mean if you are comparing against one
architecture: the small edge backbones sit well under 0.3 KB, while `vit_base`
(C=768) is several times that.

Verified end to end from a clean `git clone` on a single H200 (2026-07-29),
running exactly the commands above: **4 h 7 m** wall-clock.

**These numbers depend on which attacks you run — read this before comparing.**
The command above uses the four gradient/PGD-family attacks
(`fgsm bim pgd apgd_ce`). The committed
`results/analysis/full_eval_cifar100_summary.csv` was instead generated over the
**full six-attack suite**, which also includes DeepFool and CW, and so reports a
lower **0.940 / 0.777**. Both are correct — they answer different questions. The
per-attack means for this detector make the gap explicit:

| attack | T2 | T3 |
| --- | --- | --- |
| fgsm | 0.9986 | 0.9196 |
| bim | 0.9973 | 0.8486 |
| pgd | 0.9975 | 0.8457 |
| apgd_ce | 0.9280 | 0.7627 |
| deepfool | 0.7771 | 0.5522 |
| cw | 0.6987 | 0.5749 |

DeepFool and CW are *minimal-perturbation* attacks: they stop as soon as the
decision boundary is crossed, so the first-layer high-frequency residue Viyog
keys on is much weaker and detection is genuinely harder. Averaging them in is
what pulls 0.98 down to 0.94. To reproduce the committed **0.940 / 0.777**
instead, add `deepfool cw` to the `--attacks` list of *both*
`03_gen_adversarial.py` and `06b_extract_full.py` — expect a substantially
longer run, since both are optimisation-based and the paper caps their eval-set
size for the same reason.

Neither figure is the paper's headline **0.980 / 0.844**:
that is the full **17-architecture held-out** panel (Tier T2 — 20 architectures minus
MobileOne-S1, ResNet-18, and ResNet-34, whose archived first-layer features are
corrupted; see STATUS), whereas this core tier is a deliberately smaller,
GPU-hours-bounded 6-model subset. Reproducing the headline exactly needs the full
Tier-T2 sweep over the 17-architecture panel.

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use Viyog, cite *"Viyog: Separating Adversarial and
Out-of-Distribution"* using [CITATION.cff](CITATION.cff). It follows the
CODES+ISSS 2026 journal route in IEEE TCAD. A publication DOI, volume, issue,
and pages have not yet been assigned, so they are intentionally omitted.

For the software artifact itself, cite the immutable AE v6 archive:
[`10.5281/zenodo.21918573`](https://doi.org/10.5281/zenodo.21918573).
