Below is a revised and corrected version reflecting the key clarification:

* **Primary separation: OOD vs ADV**
* Not framed as ID vs OOD
* Testing section removed
* Language tightened accordingly

You can replace your README with this version.

---

# Viyog

<p align="center">
  <img src="Viyog.png" width="360" alt="Viyog logo"/>
</p>

**Viyog** is a lightweight, post-hoc reliability signal for convolutional neural networks (CNNs).

It extracts a simple activation-norm statistic from the **first convolutional layer** and produces a bounded score designed to help separate:

* **Out-of-Distribution (OOD)** samples
* **Adversarial (ADV)** samples

Viyog is:

* Model-agnostic
* Training-free
* Gradient-free
* Designed for post-hoc reliability analysis

It requires only forward hooks and does not modify model parameters.

---

# Purpose

OOD and adversarial samples often trigger similar alarms in primary detectors, yet arise from fundamentally different mechanisms:

* **OOD**: distributional shift
* **ADV**: structured perturbation crafted to manipulate decision boundaries

Viyog provides a lightweight secondary signal that helps distinguish between these two regimes using early-layer activation behavior.

It is intended to be used after a primary detection stage has identified suspicious inputs.

---

# Core Idea

Viyog measures the **infinity norm (max absolute activation)** of the first convolutional layer.

### Intuition

* The first convolutional layer captures low-level structural information.
* Adversarial perturbations and distributional shifts influence activation magnitudes differently.
* The maximum absolute activation provides a stable and architecture-independent scalar signal.

### Scoring Pipeline

1. Attach a forward hook to the first convolutional layer.

2. Capture per-sample activations.

3. Flatten activations per sample.

4. Compute the **infinity norm**:

   [
   |x|_\infty = \max |x_i|
   ]

5. Compute a training baseline mean via `fit()`.

6. Center new norms by this baseline.

7. Apply temperature-scaled bounded nonlinearity.

Final score range: approximately **(-1, 1)**.

Interpretation (recommended usage):

* Higher score → more likely **OOD**
* Lower score → more likely **ADV**

---

# Repository Structure

```
<repo-root>/
├── .devcontainer/
├── .github/
├── .venv/
├── .vscode/
├── docker/
├── docs/
├── src/
│   └── viyog/
│       ├── __init__.py
│       └── main.py
├── tests/
├── pyproject.toml
├── uv.lock
├── README.md
└── Viyog.png
```

### Public API

`src/viyog/__init__.py`

```python
from .main import Viyog, viyog_metrics

__all__ = ["Viyog", "viyog_metrics"]
```

All implementation resides in:

```
src/viyog/main.py
```

---

# Installation

The project uses `pyproject.toml` and `uv`.

## 1. Install uv

```bash
pip install uv
```

## 2. Clone repository

```bash
git clone <repo-url>
cd <repo-dir>
```

## 3. Create virtual environment

```bash
uv venv
```

## 4. Activate environment

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

## 5. Install package (editable mode)

```bash
uv pip install -e .
```

## 6. Run tests

```bash
pytest
```

---

# Quick Usage

```python
from viyog import Viyog, viyog_metrics

v = Viyog(model, device="cuda:0")  # or "cpu"

# Step 1: Fit baseline (required)
v.fit(train_loader)

# Step 2: Score new samples
scores = v.score(test_loader)  # torch.Tensor [N]
```

Higher score → more likely OOD
Lower score → more likely ADV

---

# Context Manager (Recommended)

```python
with Viyog(model) as v:
    v.fit(train_loader)
    ood_scores = v.score(ood_loader)
    adv_scores = v.score(adv_loader)

metrics = viyog_metrics(
    ood_scores.cpu().numpy(),
    adv_scores.cpu().numpy()
)
print(metrics)
```

Hooks are automatically removed on exit.

---

# API Reference

## `Viyog(model, device=None)`

Initializes hook and prepares scoring mechanism.

**Raises:**

* `RuntimeError` if no convolutional layer is found.

---

## `fit(train_loader)`

Computes training baseline mean of infinity norms.

Requirements:

* Must be called before `score()`
* Uses `torch.no_grad()`
* Sets `model.eval()`
* Accumulates in CPU float64 for stability

---

## `score(data_loader_or_batch)`

Returns:

```python
torch.Tensor  # shape [num_samples]
```

Raises:

* `RuntimeError` if `fit()` was not called.

---

## `close()`

Manually removes forward hook.

---

## `Viyog.Viyog_Score(centered_norms, Temperature=1000.0)`

Internal scoring transformation that converts centered norms into bounded values.

---

# Metrics

Use:

```python
viyog_metrics(ood_scores, adv_scores)
```

### Label Convention Used

| Type | Label |
| ---- | ----- |
| OOD  | 1     |
| ADV  | -1     |

Lower score → more likely ADV
Higher score → more likely OOD

---

## Returned Metrics

* **AUROC**
* **AUPR_OOD**
* **AUPR_ADV**
* **FPR95**
* **DetectionError**
* **AUTC**

  * AUFPR
  * AUFNR

These follow standard binary detection evaluation procedures.

---

# Design Constraints

* Hook attaches during initialization.
* Prefers `conv1` attribute if present.
* Otherwise selects first `Conv1d/2d/3d` layer.
* The layer must be exercised during forward pass.
* Only first element of `(inputs, labels)` batches is used.
* No gradients.
* No parameter updates.
* Stateless beyond baseline mean.

---

# Minimal Example

```python
import torch
from viyog import Viyog

model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 4, 3, padding=1),
    torch.nn.Flatten(),
    torch.nn.Linear(4 * 32 * 32, 10),
)

v = Viyog(model, device="cpu")

x = torch.randn(8, 3, 32, 32)
loader = [(x, None)]

v.fit(loader)
scores = v.score(loader)

print(scores.shape)  # torch.Size([8])

v.close()
```

---

# License

This project is licensed under the MIT License.

See the [LICENSE](LICENSE) file for full details.

# Citation

If Viyog is used in academic work, cite the associated paper or reference this repository in the methods section.
