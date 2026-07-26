"""EOT adaptive attack vs the stochastic dorm-band defense (Reviewer D-3 / C-w1).

D-3 asks for a *defense-hardening* strategy (stochastic feature sampling) characterised
under a *worst-case* adaptive attacker. ``exp_stochastic_band.py`` only established the
benign precondition (stochastic sampling does not hurt clean separation). The open gap is
the adaptive-robustness test itself: does the stochastic detector survive an attacker that
*knows* it is randomised and uses Expectation-over-Transformation (EOT) to attack the whole
band distribution at once?

Deployed stochastic defense
    Viyog reads the magnitude-normalised total variation (TV) over the dormant band.
    The hardened variant draws a RANDOM sub-band from the quiet pool *and* applies a small
    random input jitter on every inference call, then averages the TV score over ``k_def``
    draws. The band/jitter move each call, so a fixed-target attacker cannot lock on.

Attackers compared (all L-inf PGD, eps=8/255, random start)
    * ``pgd``      : plain PGD (no detector term) -- baseline.
    * ``tv_fixed`` : PGD + lambda * TV-suppression on a SINGLE fixed dorm band (the
                     deterministic Viyog* bypass). Tests whether locking one band transfers.
    * ``eot``      : PGD + lambda * EOT TV-suppression -- at every step it samples ``k_atk``
                     random (band, jitter) draws and averages the gradient over them, i.e. it
                     attacks the band distribution the defense samples from. This is the
                     VRAM-scaling worst case (k_atk forward/backward draws per step).

For each attacker we report attack-success cost and the deployed STOCHASTIC detector's
T2 (ID-vs-ADV) / T3 (OOD-vs-ADV) AUROC, plus -- for the fixed-band attacker -- the
FIXED detector's AUROC (to show it *did* break the deterministic band while the moving
band still fires).

    CUDA_VISIBLE_DEVICES=7 python experiments/exp_eot_stochastic.py --dataset cifar100 \
        --model resnet50 --n 1000 --batch 250 --steps 50 --k-atk 16 --k-def 16
"""

from __future__ import annotations

import argparse

import config
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from adaptive_attack import GradHook, auroc_dl
from config import DEVICE
from data_utils import get_id_loader, get_ood_loader
from model_utils import load_normalized_model

EPS_DEF = 8 / 255
_EPS = 1e-8


def tv_per_channel(a: torch.Tensor) -> torch.Tensor:
    """Magnitude-normalised per-channel total variation of a first-layer map.

    Args:
        a: First-conv activation, shape ``(B, C, H, W)``.

    Returns:
        Per-channel TV, shape ``(B, C)``.
    """
    fmean = a.abs().mean(dim=(2, 3))
    dh = (a[:, :, 1:, :] - a[:, :, :-1, :]).abs().mean(dim=(2, 3))
    dw = (a[:, :, :, 1:] - a[:, :, :, :-1]).abs().mean(dim=(2, 3))
    return (dh + dw) / (fmean + _EPS)


def band_means(
    model: nn.Module,
    hook: GradHook,
    x: torch.Tensor,
    bands: list[torch.Tensor],
    sigma: float,
    jitter: bool,
) -> torch.Tensor:
    """Deployed stochastic score: mean TV over each random band, averaged across draws.

    Args:
        model: Normalised classifier with a first-conv forward hook.
        hook: Hook capturing the first-conv activation.
        x: Input batch in ``[0, 1]``, shape ``(B, C, H, W)``.
        bands: List of channel-index tensors, one per draw.
        sigma: Std of the per-draw uniform input jitter (0 disables).
        jitter: Whether to apply the input jitter on each draw.

    Returns:
        Deployed score per sample, shape ``(B,)``.
    """
    outs = []
    for b in bands:
        xin = x
        if jitter and sigma > 0:
            xin = (x + (torch.rand_like(x) * 2 - 1) * sigma).clamp(0, 1)
        model(xin)
        outs.append(tv_per_channel(hook.feat)[:, b].mean(1))
    return torch.stack(outs, 1).mean(1)


@torch.no_grad()
def collect(
    model: nn.Module,
    hook: GradHook,
    loader: torch.utils.data.DataLoader,
    bands: list[torch.Tensor],
    dorm: torch.Tensor,
    sigma: float,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Forward a loader; return images, labels, stochastic score, fixed-band score."""
    xs, ys, sto, fix = [], [], [], []
    seen = 0
    for x, y in loader:
        x = x.to(DEVICE)
        sto.append(band_means(model, hook, x, bands, sigma, jitter=True).cpu().numpy())
        model(x)
        fix.append(tv_per_channel(hook.feat)[:, dorm].mean(1).cpu().numpy())
        xs.append(x.cpu())
        ys.append(y)
        seen += x.shape[0]
        if seen >= n:
            break
    return (torch.cat(xs)[:n], torch.cat(ys)[:n], np.concatenate(sto)[:n], np.concatenate(fix)[:n])


def attack(
    model: nn.Module,
    hook: GradHook,
    x: torch.Tensor,
    y: torch.Tensor,
    pool: torch.Tensor,
    dorm: torch.Tensor,
    ksz: int,
    mu: float,
    eps: float,
    alpha: float,
    steps: int,
    lam: float,
    mode: str,
    k_atk: int,
    sigma: float,
    rng: torch.Generator,
) -> torch.Tensor:
    """L-inf PGD ascending CE minus a (fixed or EOT) TV-suppression penalty.

    ``mode`` selects the attacker: ``pgd`` (no penalty), ``tv_fixed`` (suppress a single
    fixed dorm band -- the deterministic Viyog* bypass), or ``eot`` (sample ``k_atk``
    random band+jitter draws each step and average the gradient -- the adaptive worst
    case against the randomised defense). ``mu`` is the clean fixed-band TV target,
    ``pool``/``dorm``/``ksz`` define the quiet pool, deployed band and band size, and
    ``eps``/``alpha``/``steps``/``lam`` are the usual L-inf PGD knobs.

    Returns:
        Adversarial inputs, shape ``(B, C, H, W)``.
    """
    x = x.to(DEVICE)
    y = y.to(DEVICE)
    delta = (torch.rand_like(x) * 2 - 1) * eps
    delta = (x + delta).clamp(0, 1) - x
    for _ in range(steps):
        delta.requires_grad_(True)
        logits = model(x + delta)
        ce = F.cross_entropy(logits, y)
        pen = torch.zeros((), device=DEVICE)
        if mode == "tv_fixed" and lam > 0:
            tv = tv_per_channel(hook.feat)[:, dorm].mean(1)
            pen = (((tv - mu) / (mu + _EPS)) ** 2).mean()
        elif mode == "eot" and lam > 0:
            terms = []
            for _k in range(k_atk):
                idx = pool[torch.randperm(len(pool), generator=rng, device=DEVICE)[:ksz]]
                xin = (
                    (x + delta + (torch.rand_like(x) * 2 - 1) * sigma).clamp(0, 1)
                    if sigma > 0
                    else (x + delta)
                )
                model(xin)
                tv = tv_per_channel(hook.feat)[:, idx].mean(1)
                terms.append((((tv - mu) / (mu + _EPS)) ** 2).mean())
            pen = torch.stack(terms, 0).mean()
        obj = ce - lam * pen
        g = torch.autograd.grad(obj, delta)[0]
        with torch.no_grad():
            delta = (delta + alpha * g.sign()).clamp_(-eps, eps)
            delta = (x + delta).clamp(0, 1) - x
    return (x + delta).detach()


def main() -> None:
    """Run the EOT-vs-stochastic-defense robustness test for one model."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100", choices=list(config.DATASET_SPECS))
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--eps", type=float, default=EPS_DEF)
    ap.add_argument(
        "--low-pct", type=float, default=0.10, help="quiet pool size as fraction of channels"
    )
    ap.add_argument(
        "--band-pct", type=float, default=0.05, help="sub-band size as fraction of channels"
    )
    ap.add_argument("--k-atk", type=int, default=16, help="EOT draws per attack step (VRAM lever)")
    ap.add_argument("--k-def", type=int, default=16, help="defense draws averaged at inference")
    ap.add_argument(
        "--sigma", type=float, default=2 / 255, help="input jitter std for stochastic defense/EOT"
    )
    ap.add_argument("--lambdas", type=float, nargs="+", default=[10.0, 50.0, 200.0],
                    help="penalty-weight frontier for the TV-suppression attackers")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    config.set_dataset(args.dataset)

    arch = config.MODEL_ARCHS[args.model]
    wp = config.weight_path(args.dataset, args.model)
    print(
        f"=== EOT vs stochastic dorm-band [{args.dataset}/{args.model} ({arch})] "
        f"n={args.n} steps={args.steps} k_atk={args.k_atk} k_def={args.k_def} "
        f"sigma={args.sigma:.4f} ===",
        flush=True,
    )
    model = load_normalized_model(arch, wp, num_classes=config.NUM_CLASSES, device=DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    hook = GradHook(model)
    rng = torch.Generator(device=DEVICE).manual_seed(0)

    idl = get_id_loader(args.dataset, batch_size=args.batch, num_workers=4, train=False)
    xb, _ = next(iter(idl))
    with torch.no_grad():
        model(xb.to(DEVICE))
        C = hook.feat.shape[1]
    k = max(1, int(args.low_pct * C))
    ksz = max(1, int(args.band_pct * C))

    # rank live filters by mean ID activation -> quiet pool + fixed dorm band
    xs, ys, seen, means = [], [], 0, []
    for x, y in idl:
        x = x.to(DEVICE)
        with torch.no_grad():
            model(x)
            means.append(hook.feat.abs().mean(dim=(2, 3)).cpu())
        xs.append(x.cpu())
        ys.append(y)
        seen += x.shape[0]
        if seen >= args.n:
            break
    x_id = torch.cat(xs)[: args.n]
    y_id = torch.cat(ys)[: args.n]
    prof = torch.cat(means)[: args.n].mean(0)
    live = torch.where(prof > 1e-4 * prof.max())[0]
    if len(live) < max(4, ksz):
        live = torch.arange(C)
    order = live[torch.argsort(prof[live])]
    pool = order[: max(k, ksz)].to(DEVICE)  # quiet pool to sample sub-bands from
    dorm = order[:ksz].to(DEVICE)  # fixed deployed band
    def_bands = [
        pool[torch.randperm(len(pool), generator=rng, device=DEVICE)[:ksz]]
        for _ in range(args.k_def)
    ]
    mu = tv_per_channel_mu(model, hook, x_id, dorm, args.batch)
    print(f"  C={C} quiet-pool={len(pool)} band={ksz} | fixed-band TV mu={mu:.4f}", flush=True)

    # clean ID + OOD detector references (stochastic + fixed scores)
    idl2 = get_id_loader(args.dataset, batch_size=args.batch, num_workers=4, train=False)
    _, _, id_sto, id_fix = collect(model, hook, idl2, def_bands, dorm, args.sigma, args.n)
    ood_sto, ood_fix = [], []
    for oname in list(config.OOD_DATASETS)[:3]:
        try:
            ol = get_ood_loader(oname, batch_size=args.batch, num_workers=4, max_samples=args.n)
        except Exception:
            continue
        _, _, s, f = collect(model, hook, ol, def_bands, dorm, args.sigma, args.n)
        ood_sto.append(s)
        ood_fix.append(f)
        if sum(len(z) for z in ood_sto) >= args.n:
            break
    ood_sto = np.concatenate(ood_sto)
    ood_fix = np.concatenate(ood_fix)

    # clean accuracy mask
    with torch.no_grad():
        preds = [
            model(x_id[i : i + args.batch].to(DEVICE)).argmax(1).cpu()
            for i in range(0, len(x_id), args.batch)
        ]
    correct = torch.cat(preds) == y_id
    n_correct = int(correct.sum())
    print(f"  clean acc on probe set = {correct.float().mean():.3f}", flush=True)

    alpha = 2.5 * args.eps / args.steps
    print(
        f"  clean stochastic-defense reference: T2={auroc_dl(id_sto, ood_sto):.3f} "
        f"(ID-vs-OOD sanity); fixed-band T2 ref computed per-mode below",
        flush=True,
    )
    rows = []
    runs = [("pgd", 0.0)]
    for lam in args.lambdas:
        if lam <= 0:
            continue
        runs.append(("tv_fixed", lam))
        runs.append(("eot", lam))
    for mode, lam in runs:
        adv_sto, adv_fix, succ = [], [], 0
        for i in range(0, len(x_id), args.batch):
            xb = x_id[i : i + args.batch]
            yb = y_id[i : i + args.batch]
            xadv = attack(
                model,
                hook,
                xb,
                yb,
                pool,
                dorm,
                ksz,
                mu,
                args.eps,
                alpha,
                args.steps,
                lam,
                mode,
                args.k_atk,
                args.sigma,
                rng,
            )
            with torch.no_grad():
                p = model(xadv).argmax(1).cpu()
                fix = tv_per_channel(hook.feat)[:, dorm].mean(1).cpu().numpy()
                sto = (
                    band_means(model, hook, xadv, def_bands, args.sigma, jitter=True).cpu().numpy()
                )
            m = correct[i : i + xb.shape[0]]
            succ += int(((p != yb) & m).sum())
            adv_sto.append(sto)
            adv_fix.append(fix)
        adv_sto = np.concatenate(adv_sto)
        adv_fix = np.concatenate(adv_fix)
        row = {
            "mode": mode,
            "lambda": lam,
            "attack_success": round(succ / max(n_correct, 1), 3),
            "mean_tv_fixed": round(float(adv_fix.mean()), 4),
            "T2_stoch": round(auroc_dl(id_sto, adv_sto), 3),
            "T3_stoch": round(auroc_dl(ood_sto, adv_sto), 3),
            "T2_fixed": round(auroc_dl(id_fix, adv_fix), 3),
            "T3_fixed": round(auroc_dl(ood_fix, adv_fix), 3),
        }
        rows.append(row)
        print(
            f"  {mode:9} λ={lam:5.1f} | succ={row['attack_success']:.3f} "
            f"tv_fix={row['mean_tv_fixed']:.4f} | "
            f"STOCH T2/T3={row['T2_stoch']:.3f}/{row['T3_stoch']:.3f} "
            f"| FIXED T2/T3={row['T2_fixed']:.3f}/{row['T3_fixed']:.3f}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    out = args.csv or str(config.ANALYSIS_DIR / f"eot_stochastic_{args.dataset}_{args.model}.csv")
    df.to_csv(out, index=False)
    print(f"\n  saved → {out}", flush=True)
    hook.close()


@torch.no_grad()
def tv_per_channel_mu(
    model: nn.Module, hook: GradHook, x: torch.Tensor, dorm: torch.Tensor, batch: int
) -> float:
    """Mean fixed-band TV over a clean probe set (the attacker's suppression target)."""
    vals = []
    for i in range(0, len(x), batch):
        model(x[i : i + batch].to(DEVICE))
        vals.append(tv_per_channel(hook.feat)[:, dorm].mean(1).cpu())
    return float(torch.cat(vals).mean())


if __name__ == "__main__":
    main()
