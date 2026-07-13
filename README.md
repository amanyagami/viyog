# Viyog

[![PyPI](https://img.shields.io/pypi/v/viyog.svg)](https://pypi.org/project/viyog/)
[![Python](https://img.shields.io/pypi/pyversions/viyog.svg)](https://pypi.org/project/viyog/)
[![Downloads](https://static.pepy.tech/badge/viyog)](https://pepy.tech/project/viyog)
[![Docs](https://readthedocs.org/projects/viyog/badge/?version=latest)](https://viyog.readthedocs.io/en/latest/)
[![Tests](https://github.com/amanyagami/viyog/actions/workflows/test.yml/badge.svg)](https://github.com/amanyagami/viyog/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/amanyagami/viyog/blob/main/LICENSE)

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
pip install viyog                 # core: torch + numpy
pip install "viyog[metrics]"      # + scikit-learn, for viyog_metrics()
```

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

Viyog auto-detects the first conv layer (`resnet50.conv1` here), so any
`torch.nn.Module` works out of the box:

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

Across 20 architectures on CIFAR-100 (ResNet/DenseNet/ConvNeXt/Swin/ViT and edge
backbones), the dormant-band roughness score reaches **AUROC ≈ 0.966 for adversarial
detection (ID vs ADV)** and **≈ 0.824 for OOD vs ADV** — where logit detectors
(Energy/MSP/MaxLogit/GEN) are near-blind to adversarials (≤ 0.69) and feature-distance
detectors cost 4.5–40 MB of state versus Viyog's ~0.3 KB. See the accompanying paper.

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use Viyog in academic work, please cite the accompanying paper
*"Viyog: Separating Adversarial and Out-of-Distribution."*
