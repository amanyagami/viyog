"""Adaptive, norm-preserving & dormant-aware attacks — attacker-cost frontier.

This is the headline rebuttal experiment for Reviewer C-w1 (and D-3): "a white-box
attacker can bypass Viyog by adding a norm-preservation penalty to PGD." We make
that attacker concrete and measure what it actually costs.

Three attacker objectives (all L∞-bounded PGD, random start), swept over penalty
weight λ:
  * `pgd`        : plain PGD (λ ignored)             — baseline.
  * `normpresv`  : PGD + λ·((‖a₁‖∞ − μ_inf)/μ_inf)²  — holds the *L∞ Viyog score*
                   at the ID mean (the exact bypass Reviewer C describes).
  * `dormaware`  : norm-preserving + λ·dormant-fraction penalty — also tries to
                   hold the NEW dormant-band statistic at its ID mean.

For each (objective, λ) we report:
  - attack success rate (misclassification on originally-correct inputs),
  - mean ‖a₁‖∞ achieved vs the ID mean μ_inf (did the penalty hold the norm?),
  - ADV-vs-OOD AUROC for THREE detector statistics computed on the adversarial
    first-layer map: L∞ (original Viyog), dormant-fraction (new), HF-energy ratio.

Claim under test: the norm-preserving attack can flatten the L∞ AUROC, but to keep
‖a₁‖∞ fixed while still flipping the label it must inject high-frequency / dormant
energy → the dorm and HF detectors keep firing, OR the attacker pays an
attack-success cost. Either outcome is a defensible adaptive-robustness result.

    CUDA_VISIBLE_DEVICES=1 python experiments/adaptive_attack.py --dataset cifar100 \
        --model resnet50 --n 512 --steps 30 --lambdas 0 1 3 10 30 100
"""
from __future__ import annotations

import argparse

import config
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DEVICE
from data_utils import get_id_loader, get_ood_loader
from model_utils import find_first_conv_in_normalized, load_normalized_model
from sklearn.metrics import roc_auc_score

EPS_DEF = 8 / 255
_EPS = 1e-8


class GradHook:
    """Capture the first-conv output WITH the autograd graph intact."""
    def __init__(self, norm_model: nn.Module) -> None:
        _, layer = find_first_conv_in_normalized(norm_model)
        self.feat: torch.Tensor | None = None
        self.h = layer.register_forward_hook(lambda m, i, o: setattr(self, "feat", o))

    def close(self) -> None:
        self.h.remove()


def stats_from_feat(a: torch.Tensor, dorm_idx: torch.Tensor) -> dict:
    """Per-sample detector statistics from a (B,C,H,W) first-layer map."""
    absa = a.abs()
    infn = absa.flatten(1).amax(1)                                  # (B,)
    fmean = absa.mean(dim=(2, 3))                                   # (B,C)
    dorm = fmean[:, dorm_idx].sum(1) / (fmean.sum(1) + _EPS)        # (B,)
    blur = F.avg_pool2d(a, 3, 1, 1)
    hf = (a - blur).pow(2).mean(dim=(1, 2, 3)) / (a.pow(2).mean(dim=(1, 2, 3)) + _EPS)
    # tv|dorm: the DEPLOYED Viyog* score -- mean magnitude-normalised total
    # variation over the dormant band (so the attacker can target the real channel).
    dh = (a[:, :, 1:, :] - a[:, :, :-1, :]).abs().mean(dim=(2, 3))   # (B,C)
    dw = (a[:, :, :, 1:] - a[:, :, :, :-1]).abs().mean(dim=(2, 3))   # (B,C)
    tv = (dh + dw) / (fmean + _EPS)                                  # (B,C) magnitude-normalised
    tvdorm = tv[:, dorm_idx].mean(1)                                 # (B,)
    return {"linf": infn, "dorm": dorm, "hf": hf, "tvdorm": tvdorm}


def auroc_dl(neg: np.ndarray, pos: np.ndarray) -> float:
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    s = np.r_[neg, pos]
    if len(np.unique(s)) < 2:
        return 0.5
    a = roc_auc_score(y, s)
    return float(max(a, 1 - a))


@torch.no_grad()
def collect_clean(model, hook, loader, dorm_idx, n):
    """Forward clean inputs; return stacked detector stats + images/labels."""
    xs, ys, st = [], [], {"linf": [], "dorm": [], "hf": [], "tvdorm": []}
    seen = 0
    for x, y in loader:
        x = x.to(DEVICE)
        model(x)
        s = stats_from_feat(hook.feat, dorm_idx)
        for k in st:
            st[k].append(s[k].cpu())
        xs.append(x.cpu()); ys.append(y)
        seen += x.shape[0]
        if seen >= n:
            break
    st = {k: torch.cat(v)[:n].numpy() for k, v in st.items()}
    return torch.cat(xs)[:n], torch.cat(ys)[:n], st


def adaptive_pgd(model, hook, x, y, dorm_idx, mu_inf, mu_dorm, mu_hf, mu_tvdorm, eps, alpha, steps,
                 lam, mode):
    """L∞ PGD ascending CE minus a detector-evasion penalty (random start).

    Penalty targets, by mode (all hold the L∞ Viyog score where applicable):
      normpresv : ‖a₁‖∞ only          dormaware : ‖a₁‖∞ + dorm-band fraction
      hfaware   : ‖a₁‖∞ + dorm-band HF tvaware   : ‖a₁‖∞ + dorm-band TV (deployed Viyog*)
      allaware  : ‖a₁‖∞ + dorm frac + dorm HF + dorm TV (worst case)
    """
    x = x.to(DEVICE); y = y.to(DEVICE)
    delta = (torch.rand_like(x) * 2 - 1) * eps
    delta = (x + delta).clamp(0, 1) - x
    for _ in range(steps):
        delta.requires_grad_(True)
        logits = model(x + delta)
        s = stats_from_feat(hook.feat, dorm_idx)
        ce = F.cross_entropy(logits, y)
        pen = torch.zeros((), device=DEVICE)
        if mode in ("normpresv", "dormaware", "hfaware", "tvaware", "allaware") and lam > 0:
            pen = pen + (((s["linf"] - mu_inf) / (mu_inf + _EPS)) ** 2).mean()
        if mode in ("dormaware", "allaware") and lam > 0:
            pen = pen + (((s["dorm"] - mu_dorm) / (mu_dorm + _EPS)) ** 2).mean()
        if mode in ("hfaware", "allaware") and lam > 0:
            pen = pen + (((s["hf"] - mu_hf) / (mu_hf + _EPS)) ** 2).mean()
        if mode in ("tvaware", "allaware") and lam > 0:
            pen = pen + (((s["tvdorm"] - mu_tvdorm) / (mu_tvdorm + _EPS)) ** 2).mean()
        obj = ce - lam * pen                       # maximize CE, minimize penalty
        g = torch.autograd.grad(obj, delta)[0]
        with torch.no_grad():
            delta = (delta + alpha * g.sign()).clamp_(-eps, eps)
            delta = (x + delta).clamp(0, 1) - x
    return (x + delta).detach()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100", choices=list(config.DATASET_SPECS))
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--batch", type=int, default=128,
                    help="GPU chunk size for clean/attack passes (bigger = more VRAM, faster on free GPUs)")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--eps", type=float, default=EPS_DEF)
    ap.add_argument("--low-pct", type=float, default=0.10)
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0, 1, 3, 10, 30, 100])
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    config.set_dataset(args.dataset)

    arch = config.MODEL_ARCHS[args.model]
    wp = config.weight_path(args.dataset, args.model)
    print(f"=== adaptive attack [{args.dataset}/{args.model} ({arch})] "
          f"n={args.n} steps={args.steps} eps={args.eps:.4f} ===", flush=True)
    model = load_normalized_model(arch, wp, num_classes=config.NUM_CLASSES, device=DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    hook = GradHook(model)

    # ID: fit dormant ranking + statistic means; OOD: detector negatives
    idl = get_id_loader(args.dataset, batch_size=128, num_workers=4, train=False)
    # provisional dorm_idx from first clean batch, then refine over n
    xb, yb = next(iter(idl))
    with torch.no_grad():
        model(xb.to(DEVICE))
        C = hook.feat.shape[1]
    k = max(1, int(args.low_pct * C))
    # rank filters by mean ID activation over the sample
    x_id, y_id, _ = collect_clean(model, hook, idl, torch.arange(k, device=DEVICE), args.n)
    with torch.no_grad():
        means = []
        for i in range(0, len(x_id), args.batch):
            model(x_id[i:i + args.batch].to(DEVICE))
            means.append(hook.feat.abs().mean(dim=(2, 3)).cpu())
        fmean_id = torch.cat(means).mean(0)
    dorm_idx = torch.argsort(fmean_id)[:k].to(DEVICE)

    # recompute clean ID stats with the real dorm ranking
    _, _, id_st = collect_clean(model, hook, idl, dorm_idx, args.n)
    mu_inf = float(np.mean(id_st["linf"]))
    mu_dorm = float(np.mean(id_st["dorm"]))
    mu_hf = float(np.mean(id_st["hf"]))
    mu_tvdorm = float(np.mean(id_st["tvdorm"]))
    print(f"  ID means: ‖a₁‖∞ μ={mu_inf:.3f} | dorm μ={mu_dorm:.4f} | hf μ={mu_hf:.4f} | tv|dorm μ={mu_tvdorm:.4f}", flush=True)

    # OOD detector negatives (pool a couple of OOD sets)
    ood_st = {"linf": [], "dorm": [], "hf": [], "tvdorm": []}
    got = 0
    for oname in list(config.OOD_DATASETS)[:3]:
        try:
            ol = get_ood_loader(oname, batch_size=128, num_workers=4, max_samples=args.n)
        except Exception:  # noqa: BLE001
            continue
        _, _, st = collect_clean(model, hook, ol, dorm_idx, args.n)
        for kk in ood_st:
            ood_st[kk].append(st[kk])
        got += len(st["linf"])
        if got >= args.n:
            break
    ood_st = {k: np.concatenate(v) for k, v in ood_st.items()}

    # clean accuracy mask (attack success measured on originally-correct inputs)
    with torch.no_grad():
        preds = []
        for i in range(0, len(x_id), args.batch):
            preds.append(model(x_id[i:i + args.batch].to(DEVICE)).argmax(1).cpu())
        clean_pred = torch.cat(preds)
    correct = (clean_pred == y_id)
    print(f"  clean acc on probe set = {correct.float().mean():.3f}", flush=True)

    alpha = 2.5 * args.eps / args.steps
    rows = []
    modes = [("pgd", [0.0])] + [(m, args.lambdas)
                                for m in ("normpresv", "dormaware", "hfaware", "tvaware", "allaware")]
    import pandas as pd
    for mode, lams in modes:
        for lam in lams:
            advs, st = [], {"linf": [], "dorm": [], "hf": [], "tvdorm": []}
            succ = 0
            for i in range(0, len(x_id), args.batch):
                xb = x_id[i:i + args.batch]; yb = y_id[i:i + args.batch]
                xadv = adaptive_pgd(model, hook, xb, yb, dorm_idx, mu_inf, mu_dorm, mu_hf, mu_tvdorm,
                                    args.eps, alpha, args.steps, lam, mode)
                with torch.no_grad():
                    p = model(xadv).argmax(1).cpu()
                    s = stats_from_feat(hook.feat, dorm_idx)
                m = correct[i:i + args.batch]
                succ += int(((p != yb) & m).sum())
                for kk in st:
                    st[kk].append(s[kk].cpu().numpy())
                advs.append(xadv.cpu())
            st = {k: np.concatenate(v) for k, v in st.items()}
            n_correct = int(correct.sum())
            sr = succ / max(n_correct, 1)
            row = {
                "mode": mode, "lambda": lam,
                "attack_success": round(sr, 3),
                "mean_linf": round(float(st["linf"].mean()), 3),
                "linf_dev_from_ID_%": round(100 * (st["linf"].mean() - mu_inf) / mu_inf, 1),
                "auroc_linf": round(auroc_dl(ood_st["linf"], st["linf"]), 3),
                "auroc_dorm": round(auroc_dl(ood_st["dorm"], st["dorm"]), 3),
                "auroc_hf": round(auroc_dl(ood_st["hf"], st["hf"]), 3),
                "auroc_tvdorm": round(auroc_dl(ood_st["tvdorm"], st["tvdorm"]), 3),
            }
            rows.append(row)
            print(f"  {mode:10} λ={lam:6.1f} | succ={sr:.3f} "
                  f"‖∞‖={row['mean_linf']:.3f}({row['linf_dev_from_ID_%']:+.0f}%) "
                  f"AUROC linf/dorm/hf/tv={row['auroc_linf']:.3f}/{row['auroc_dorm']:.3f}/"
                  f"{row['auroc_hf']:.3f}/{row['auroc_tvdorm']:.3f}",
                  flush=True)

    df = pd.DataFrame(rows)
    out = args.csv or str(config.ANALYSIS_DIR / f"adaptive_{args.dataset}_{args.model}.csv")
    df.to_csv(out, index=False)
    print(f"\n  saved → {out}", flush=True)
    hook.close()


if __name__ == "__main__":
    main()
