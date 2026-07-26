"""Attack-budget sweep: does first-layer ||.||_inf suppression appear at larger eps?

The paper's mechanism claims ||ADV||_inf < ||ID||_inf. Our 8/255 PGD/FGSM samples
show NO suppression (ADV ~= ID). This sweeps the L-inf budget eps in {4,8,16,32}/255
(PGD, random start) and reports, per model: mean ||a1||_inf for ID vs ADV vs OOD,
the suppression % (ADV vs ID), and the raw-inf-norm OOD-vs-ADV (T3) and ID-vs-ADV
(T2) AUROC at each eps. If suppression + separation grow with eps, the paper's
mechanism is real but budget-dependent; if not, it does not hold for gradient L-inf
attacks at any standard budget.

    CUDA_VISIBLE_DEVICES=1 python experiments/eps_sweep.py --dataset cifar100 \
        --models resnet50 densenet121 convnextv2_base mobilenetv3_l --n 512 --steps 20
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


class GradHook:
    def __init__(self, m: nn.Module) -> None:
        _, layer = find_first_conv_in_normalized(m)
        self.feat = None
        self.h = layer.register_forward_hook(lambda mod, i, o: setattr(self, "feat", o))

    def close(self) -> None:
        self.h.remove()


def auroc_dl(neg, pos):
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    s = np.r_[neg, pos]
    if len(np.unique(s)) < 2:
        return 0.5
    return float(max(roc_auc_score(y, s), 1 - roc_auc_score(y, s)))


def pgd(model, hook, x, y, eps, steps):
    x = x.to(DEVICE); y = y.to(DEVICE)
    alpha = 2.5 * eps / steps
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


@torch.no_grad()
def inf_norms(model, hook, x):
    model(x.to(DEVICE))
    return hook.feat.abs().flatten(1).amax(1).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100", choices=list(config.DATASET_SPECS))
    ap.add_argument("--models", nargs="+", default=["resnet50", "densenet121", "convnextv2_base", "mobilenetv3_l"])
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--eps", type=float, nargs="+", default=[4 / 255, 8 / 255, 16 / 255, 32 / 255])
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    config.set_dataset(args.dataset)
    import pandas as pd

    rows = []
    for m in args.models:
        arch = config.MODEL_ARCHS[m]
        wp = config.weight_path(args.dataset, m)
        if not wp.exists():
            print(f"[skip] {m}: no weights"); continue
        model = load_normalized_model(arch, wp, num_classes=config.NUM_CLASSES, device=DEVICE).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        hook = GradHook(model)
        print(f"\n=== {m} ({arch}) ===", flush=True)

        # collect ID images + ID inf-norms
        idl = get_id_loader(args.dataset, batch_size=128, num_workers=4, train=False)
        xs, ys, idn = [], [], []
        for x, y in idl:
            idn.append(inf_norms(model, hook, x)); xs.append(x); ys.append(y)
            if sum(len(t) for t in xs) >= args.n:
                break
        X = torch.cat(xs)[:args.n]; Y = torch.cat(ys)[:args.n]; idn = np.concatenate(idn)[:args.n]
        # OOD inf-norms (first 3 sets)
        oodn = []
        for o in list(config.OOD_DATASETS)[:3]:
            try:
                ol = get_ood_loader(o, batch_size=128, num_workers=4, max_samples=args.n // 3 + 1)
            except Exception:  # noqa: BLE001
                continue
            for batch in ol:
                xb = batch[0] if isinstance(batch, (list, tuple)) else batch
                oodn.append(inf_norms(model, hook, xb))
        oodn = np.concatenate(oodn)
        mu_id = idn.mean()
        print(f"  ID ||inf|| mu={mu_id:.3f} | OOD mu={oodn.mean():.3f}", flush=True)

        for eps in args.eps:
            advn = []
            for i in range(0, len(X), 128):
                xadv = pgd(model, hook, X[i:i + 128], Y[i:i + 128], eps, args.steps)
                advn.append(inf_norms(model, hook, xadv))
            advn = np.concatenate(advn)
            supp = 100 * (advn.mean() - mu_id) / mu_id
            t2 = auroc_dl(idn, advn)
            t3 = auroc_dl(oodn, advn)
            rows.append({"model": m, "eps_255": round(eps * 255, 1),
                         "ID_inf": round(mu_id, 3), "ADV_inf": round(float(advn.mean()), 3),
                         "OOD_inf": round(float(oodn.mean()), 3),
                         "ADV_vs_ID_%": round(supp, 2),
                         "rawinf_T2_ID_ADV": round(t2, 3), "rawinf_T3_OOD_ADV": round(t3, 3)})
            print(f"  eps={eps*255:4.0f}/255 | ADV||inf||={advn.mean():.3f} ({supp:+.1f}% vs ID) "
                  f"| rawinf T2={t2:.3f} T3={t3:.3f}", flush=True)
        hook.close(); del model; torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    out = args.csv or str(config.ANALYSIS_DIR / f"eps_sweep_{args.dataset}.csv")
    df.to_csv(out, index=False)
    print(f"\n  saved → {out}", flush=True)
    print("\n=== suppression by eps (mean over models) ===")
    print(df.groupby("eps_255")[["ADV_vs_ID_%", "rawinf_T2_ID_ADV", "rawinf_T3_OOD_ADV"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
