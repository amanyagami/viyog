r"""Non-vision sanity check for the dormant-band roughness statistic (closes D-w3).

Reviewer D-w3 asked whether the method generalises beyond images. The mechanism of
Prop. ``prop:shape`` is modality-agnostic: an adversarial perturbation injects
broadband high-frequency content that raises the per-channel total variation (TV)
of the \\emph{quietest} first-layer channels, where the natural signal is smooth.
Nothing in that argument is 2D-specific. This script tests it on a controlled
\\emph{1D signal} task with a 1D-CNN:

  * ID: K classes of band-limited multi-sinusoid signals (+ noise);
  * OOD: signals whose frequencies are shifted out of the ID band (near-OOD);
  * ADV: FGSM / PGD (L-inf) attacks on the trained 1D-CNN.

It then reads the deployed statistic V(x) on the FIRST Conv1d layer -- the
dormant-band (bottom-10% of active channels) average of TV(a_c)/(mean|a_c|+eps),
the exact 1D analogue of Eq. vdef -- and reports ID-vs-ADV (T2) and OOD-vs-ADV
(T3) AUROC for V and, for contrast, the raw L-inf. If V separates ADV from ID/OOD
on 1D as it does on images, the shape mechanism is not vision-specific.

Self-contained (synthetic data, CPU): no download, no GPU, fully reproducible.

    python experiments/exp_nonvision.py --csv results/analysis/nonvision_1d.csv
"""

from __future__ import annotations

import argparse

import config
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

EPS = 1e-6
KAPPA = 1e-4
DORM_P = 0.10
SEQ_LEN = 256
N_CLASSES = 10


def make_signals(
    n: int, rng: np.random.Generator, freq_lo: int, freq_hi: int
) -> tuple[np.ndarray, np.ndarray]:
    """K-class band-limited multi-sinusoid signals with per-class base frequency."""
    t = np.linspace(0, 1, SEQ_LEN, dtype=np.float32)
    freqs = np.linspace(freq_lo, freq_hi, N_CLASSES)
    x = np.empty((n, SEQ_LEN), dtype=np.float32)
    y = rng.integers(0, N_CLASSES, size=n)
    for i in range(n):
        f = freqs[y[i]]
        # smooth band-limited signal (low natural TV) + a small amount of noise;
        # closely-spaced class frequencies keep the task non-trivial so attacks land
        sig = np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        sig += 0.3 * np.sin(2 * np.pi * (1.5 * f) * t + rng.uniform(0, 2 * np.pi))
        sig += 0.06 * rng.standard_normal(SEQ_LEN).astype(np.float32)
        x[i] = sig
    x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + EPS)
    return x, y


class Net1D(nn.Module):
    """Small 1D-CNN; the detector reads the first conv (conv1)."""

    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, channels, 7, padding=3)
        self.conv2 = nn.Conv1d(channels, 64, 5, padding=2)
        self.conv3 = nn.Conv1d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(64, N_CLASSES)

    def first(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv1(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pool(F.relu(self.conv1(x)))
        h = self.pool(F.relu(self.conv2(h)))
        h = F.relu(self.conv3(h))
        return self.fc(h.mean(dim=2))


@torch.no_grad()
def shape_linf(model: Net1D, x: torch.Tensor, dorm: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample V(x) (dorm-band TV/mean) and raw L-inf at the first Conv1d."""
    a = model.first(x)  # (B,C,L)
    mean_abs = a.abs().mean(dim=2)  # (B,C)
    tv = (a[:, :, 1:] - a[:, :, :-1]).abs().mean(dim=2)  # (B,C)
    shape = tv / (mean_abs + EPS)
    v = shape.index_select(1, dorm).mean(dim=1)
    linf = a.abs().amax(dim=(1, 2))
    return v.cpu().numpy(), linf.cpu().numpy()


@torch.no_grad()
def profile_dorm(model: Net1D, x: torch.Tensor) -> torch.Tensor:
    """Bottom-10% of active first-conv channels by ID-mean activation."""
    a = model.first(x)
    prof = a.abs().mean(dim=(0, 2)).cpu().numpy()  # (C,)
    alive = np.where(prof > KAPPA)[0]
    if len(alive) == 0:
        alive = np.arange(len(prof))
    k = max(1, round(DORM_P * len(alive)))
    order = alive[np.argsort(prof[alive])]
    return torch.as_tensor(order[:k], dtype=torch.long)


def fgsm(model: Net1D, x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.clone().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    (g,) = torch.autograd.grad(loss, x)
    return (x + eps * g.sign()).detach()


def pgd(
    model: Net1D, x: torch.Tensor, y: torch.Tensor, eps: float, steps: int = 20
) -> torch.Tensor:
    x0 = x.clone()
    xa = x.clone() + torch.empty_like(x).uniform_(-eps, eps)
    alpha = 2.5 * eps / steps
    for _ in range(steps):
        xa = xa.detach().requires_grad_(True)
        loss = F.cross_entropy(model(xa), y)
        (g,) = torch.autograd.grad(loss, xa)
        xa = xa + alpha * g.sign()
        xa = torch.clamp(xa, x0 - eps, x0 + eps)
    return xa.detach()


def dauroc(id_s: np.ndarray, ood_s: np.ndarray) -> float:
    """Directionless AUROC = max(a, 1-a)."""
    y = np.concatenate([np.zeros(len(id_s)), np.ones(len(ood_s))])
    s = np.concatenate([id_s, ood_s])
    a = roc_auc_score(y, s)
    return round(float(max(a, 1 - a)), 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=1500)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.2, 0.4])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = "cpu"

    xtr, ytr = make_signals(args.n_train, rng, 3, 12)
    xte, yte = make_signals(args.n_test, rng, 3, 12)
    xood, _ = make_signals(args.n_test, rng, 16, 28)  # near-OOD: shifted frequency band
    to = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev).unsqueeze(1)  # noqa: E731
    Xtr, Ytr = to(xtr), torch.as_tensor(ytr, device=dev)
    Xte, Yte = to(xte), torch.as_tensor(yte, device=dev)
    Xood = to(xood)

    model = Net1D().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 256):
            idx = perm[i : i + 256]
            opt.zero_grad()
            loss = F.cross_entropy(model(Xtr[idx]), Ytr[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(Xte).argmax(1) == Yte).float().mean().item()
    print(f"1D-CNN clean test acc = {acc:.3f}", flush=True)

    dorm = profile_dorm(model, Xtr)
    print(f"first conv: {model.conv1.out_channels} ch, dorm band = {len(dorm)} ch {dorm.tolist()}")

    v_id, l_id = shape_linf(model, Xte, dorm)
    v_ood, l_ood = shape_linf(model, Xood, dorm)

    rows = []
    for eps in args.eps:
        xf = fgsm(model, Xte, Yte, eps)
        xp = pgd(model, Xte, Yte, eps)
        for atk, xa in [("fgsm", xf), ("pgd", xp)]:
            with torch.no_grad():
                succ = (model(xa).argmax(1) != Yte).float().mean().item()
            v_a, l_a = shape_linf(model, xa, dorm)
            row = {
                "attack": atk,
                "eps": eps,
                "succ": round(succ, 3),
                "T2_shape": dauroc(v_id, v_a),
                "T2_linf": dauroc(l_id, l_a),
                "T3_shape": dauroc(v_ood, v_a),
                "T3_linf": dauroc(l_ood, l_a),
            }
            rows.append(row)
            print(
                f"  {atk:5} eps={eps:.3f} succ={succ:.2f} | "
                f"T2 shape={row['T2_shape']:.3f} linf={row['T2_linf']:.3f} | "
                f"T3 shape={row['T3_shape']:.3f} linf={row['T3_linf']:.3f}",
                flush=True,
            )

    import pandas as pd

    df = pd.DataFrame(rows)
    out = args.csv or str(config.ANALYSIS_DIR / "nonvision_1d.csv")
    df.to_csv(out, index=False)
    print(
        f"\nMEAN T2 shape={df.T2_shape.mean():.3f} (linf {df.T2_linf.mean():.3f}) | "
        f"T3 shape={df.T3_shape.mean():.3f} (linf {df.T3_linf.mean():.3f})"
    )
    print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
