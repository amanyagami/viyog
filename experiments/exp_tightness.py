"""Measured tightness of the Proposition-1 bound ||e_m||_inf <= ||W_m||_1 * eps (Reviewer B-4).

B-4 objects that Prop. 1 is an *upper bound* whose practical tightness is never analysed.
We make it concrete: for each backbone we measure the realised first-layer perturbation
norm ||e_m||_inf = ||f0(x+delta) - f0(x)||_inf under FGSM and PGD, and the induced operator
constant ||W_m||_1 (max over output channels of the per-filter L1 weight sum -- the
infinity-norm of the Toeplitz/patch-embed operator). The dimensionless ratio

    tau = ||e_m||_inf / (||W_m||_1 * eps)  in (0, 1]

quantifies how close the attack drives the first layer to the worst case. tau near 1 means
the bound is tight (the attacker saturates it); tau small means the gradient-sign direction
is far from the operator's worst-case singular direction -- i.e. the suppression is real but
the bound is loose, which is the honest nuance B-4 asks for.

    CUDA_VISIBLE_DEVICES=6 python experiments/exp_tightness.py --dataset cifar100 \
        --models resnet50 densenet121 mobilenetv3_l swin_tiny convnextv2_base vit_base
"""

from __future__ import annotations

import argparse

import config
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from config import DEVICE
from data_utils import get_id_loader
from model_utils import find_first_conv_in_normalized, load_normalized_model

EPS_DEF = 8 / 255


class IOHook:
    """Capture the first-conv input *and* output (input is post-normalisation).

    The bound in Prop. 1 is on the perturbation that actually reaches the conv,
    ``||delta_in||_inf``, which after the folded input normalisation is ~eps/sigma, not
    the raw image eps. Measuring it directly keeps tau in ``(0, 1]``.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        _, layer = find_first_conv_in_normalized(model)
        self.inp: torch.Tensor | None = None
        self.out: torch.Tensor | None = None
        self.h1 = layer.register_forward_pre_hook(lambda m, i: setattr(self, "inp", i[0]))
        self.h2 = layer.register_forward_hook(lambda m, i, o: setattr(self, "out", o))

    def close(self) -> None:
        self.h1.remove()
        self.h2.remove()


def first_layer_operator_1norm(model: torch.nn.Module) -> float:
    """Induced infinity-norm of the first linear layer (= ``||W_m||_1`` in Prop. 1).

    For a conv this is the max over output channels of the per-filter L1 weight sum
    (the max absolute row sum of the equivalent Toeplitz matrix); for a patch-embed
    projection it is the same statistic over the linear weight.

    Args:
        model: Normalised classifier.

    Returns:
        The operator constant ``||W_m||_1``.
    """
    _, layer = find_first_conv_in_normalized(model)
    w = layer.weight.detach()
    return float(w.abs().flatten(1).sum(1).max())


def pgd(
    model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float, alpha: float, steps: int
) -> torch.Tensor:
    """Standard L-inf PGD (random start) returning adversarial inputs."""
    delta = (torch.rand_like(x) * 2 - 1) * eps
    delta = (x + delta).clamp(0, 1) - x
    for _ in range(steps):
        delta.requires_grad_(True)
        loss = F.cross_entropy(model(x + delta), y)
        g = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            delta = (delta + alpha * g.sign()).clamp_(-eps, eps)
            delta = (x + delta).clamp(0, 1) - x
    return (x + delta).detach()


def main() -> None:
    """Measure tau for each backbone under FGSM and PGD and write a CSV."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100", choices=list(config.DATASET_SPECS))
    ap.add_argument("--models", nargs="+", default=["resnet50"])
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--eps", type=float, default=EPS_DEF)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    config.set_dataset(args.dataset)
    alpha = 2.5 * args.eps / args.steps
    rows = []

    for mname in args.models:
        arch = config.MODEL_ARCHS[mname]
        wp = config.weight_path(args.dataset, mname)
        model = load_normalized_model(
            arch, wp, num_classes=config.NUM_CLASSES, device=DEVICE
        ).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        hook = IOHook(model)
        wm1 = first_layer_operator_1norm(model)

        idl = get_id_loader(args.dataset, batch_size=args.batch, num_workers=4, train=False)
        for atk in ("fgsm", "pgd"):
            em, din, seen = [], [], 0
            for x, y in idl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                with torch.no_grad():
                    model(x)
                    in_c = hook.inp.detach().clone()
                    out_c = hook.out.detach().clone()
                steps = 1 if atk == "fgsm" else args.steps
                a = args.eps if atk == "fgsm" else alpha
                xadv = pgd(model, x, y, args.eps, a, steps)
                with torch.no_grad():
                    model(xadv)
                    d = (hook.inp - in_c).abs().flatten(1).amax(1)  # ||delta_in||_inf at conv input
                    e = (hook.out - out_c).abs().flatten(1).amax(1)  # ||e_m||_inf per sample
                em.append(e.cpu().numpy())
                din.append(d.cpu().numpy())
                seen += x.shape[0]
                if seen >= args.n:
                    break
            em = np.concatenate(em)[: args.n]
            din = np.concatenate(din)[: args.n]
            tau = em / (wm1 * din + 1e-12)  # realised / worst-case operator response, in (0,1]
            rows.append(
                {
                    "model": mname,
                    "attack": atk,
                    "Wm_1": round(wm1, 4),
                    "din_inf_mean": round(float(din.mean()), 4),
                    "em_inf_mean": round(float(em.mean()), 4),
                    "em_inf_p95": round(float(np.percentile(em, 95)), 4),
                    "tau_mean": round(float(tau.mean()), 4),
                    "tau_p50": round(float(np.percentile(tau, 50)), 4),
                    "tau_p95": round(float(np.percentile(tau, 95)), 4),
                }
            )
            print(
                f"  {mname:16} {atk:4} | ||W_m||_1={wm1:.3f} ||delta_in||={din.mean():.3f} "
                f"||e_m||_inf={em.mean():.4f} | tau_mean={tau.mean():.4f} "
                f"p50={np.percentile(tau, 50):.4f} p95={np.percentile(tau, 95):.4f}",
                flush=True,
            )
        hook.close()
        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    out = args.csv or str(config.ANALYSIS_DIR / f"tightness_{args.dataset}.csv")
    df.to_csv(out, index=False)
    print("\n=== tau summary (mean over models) ===")
    print(df.groupby("attack")[["tau_mean", "tau_p95"]].mean().round(4).to_string())
    print(f"\n  saved → {out}", flush=True)


if __name__ == "__main__":
    main()
