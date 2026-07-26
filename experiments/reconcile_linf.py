"""P0 reconciliation: does L-inf separate L2-attack ADV (CW/DeepFool) from OOD
far better than the L-inf attacks (fgsm/bim/pgd/apgd_ce) we extracted?

Extracts per-sample first-conv L-inf from the existing CW/DeepFool adv samples
(small batch, forward-only), compares OOD-vs-ADV AUROC by attack type. This tests
whether the paper's 92.38 headline is driven by the L2/minimal-norm attack mix.
"""
from __future__ import annotations
import glob, os, sys
import h5py, numpy as np, torch
from sklearn.metrics import roc_auc_score

import config
from data_utils import adv_loader_from_h5
from model_utils import FirstLayerHook, load_normalized_model

config.set_dataset("cifar100")
FD = config.FEATURES_DIR
DEV = config.DEVICE
MODELS = sys.argv[1:] or ["convnextv2_base", "vit_base", "swin_tiny"]


def dl(a, b):
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]; s = np.r_[a, b]
    if len(np.unique(s)) < 2:
        return 0.5
    v = roc_auc_score(y, s); return max(v, 1 - v)


@torch.no_grad()
def inf_from_samples(arch, model, attack):
    """Per-sample first-conv L-inf for an adv-sample h5, batch 64."""
    wp = config.weight_path("cifar100", model)
    nm = load_normalized_model(arch, wp if wp.exists() else None,
                               num_classes=config.NUM_CLASSES, device=DEV)
    h5 = config.ADV_DIR / f"{model}_{attack}.h5"
    if not h5.exists():
        return None
    out = []
    with FirstLayerHook(nm) as hook:
        for imgs, _ in adv_loader_from_h5(h5, batch_size=64, num_workers=2):
            nm(imgs.to(DEV))
            f = hook.features                       # (B,C,H,W)
            out.append(f.abs().flatten(1).amax(1).cpu().numpy())
    del nm; torch.cuda.empty_cache()
    return np.concatenate(out).astype(np.float64)


def ood_inf(model):
    return [h5py.File(p)["inf_norms"][:].astype(np.float64)
            for p in glob.glob(str(FD / f"featfull_{model}_ood_*.h5"))]


for model in MODELS:
    arch = config.MODEL_ARCHS[model]
    oods = ood_inf(model)
    if not oods:
        print(f"{model}: no OOD features"); continue
    o_all = np.concatenate(oods)
    row = {}
    # existing L-inf attacks (from featfull)
    for p in glob.glob(str(FD / f"featfull_{model}_adv_*.h5")):
        atk = os.path.basename(p).split("_adv_")[1][:-3]
        a = h5py.File(p)["inf_norms"][:].astype(np.float64)
        row[atk] = (dl(o_all, a), np.mean([dl(o, a) for o in oods]))
    # L2 attacks (extract now)
    for atk in ("cw", "deepfool"):
        a = inf_from_samples(arch, model, atk)
        if a is not None:
            row[atk] = (dl(o_all, a), np.mean([dl(o, a) for o in oods]))
    print(f"\n{model}  (OOD-vs-ADV L-inf AUROC: pooled / per-cell)")
    for atk in sorted(row):
        p, c = row[atk]
        tag = "  <-- L2" if atk in ("cw", "deepfool") else ""
        print(f"    {atk:10} pooled={p:.3f}  per-cell={c:.3f}{tag}")
