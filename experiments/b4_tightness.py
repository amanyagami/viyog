"""B4 tightness: measured tau = mean ||e_m||_inf / (||W_m||_1 * eps_norm).

For each backbone, e_m is the FIRST-CONV output difference between a clean ID
image and its matched (same-order) adversarial image. The first conv sees the
NORMALIZED input (x-mean)/std, so the effective perturbation entering the conv
is r/std and the operator bound becomes
    ||e_m||_inf <= ||W_m||_1 * (eps / min(std)).
We report:
  - W1            : ||W_m||_1 = max over out-channels of sum |kernel| (the
                    induced-inf operator norm of the conv on normalized input,
                    folding the per-input-channel 1/std scaling into the kernel).
  - eps_norm      : eps / min(std)  (worst-case ||r||_inf in normalized space)
  - bound         : W1 * eps_norm   (RHS of Prop.1 eq:pertbound)
  - emax_mean/95/max : statistics of per-sample ||e_m||_inf
  - tau_mean/95/max  : emax / bound
Run from experiments/ dir.
"""
from __future__ import annotations
import sys
import numpy as np
import torch
import config
from model_utils import FirstLayerHook, find_first_conv_in_normalized, load_normalized_model
from data_utils import get_id_loader, adv_loader_from_h5

config.set_dataset("cifar100")
DEVICE = config.DEVICE
EPS = 8 / 255
STD = config.IMAGENET_STD
MIN_STD = min(STD)
EPS_NORM = EPS / MIN_STD

MODELS = sys.argv[1].split(",") if len(sys.argv) > 1 else [
    "resnet50", "convnextv2_base", "densenet121", "vit_base", "swin_tiny"]
ATTACKS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["fgsm", "pgd"]
N = 1000  # samples; enough for stable mean/percentiles, fast


def w1_norm(layer) -> float:
    """||W_m||_1 = induced inf-norm of the first conv on NORMALIZED input.

    Conv weight W has shape (Cout, Cin, kh, kw). The conv acts on x_norm =
    (x-mean)/std, so per input channel c the effective weight is W[:,c]/std[c].
    The induced inf operator norm of a conv (max abs row sum of its Toeplitz
    matrix) = max over out-channels of the total absolute kernel weight.
    """
    W = layer.weight.detach().float().cpu()  # (Cout,Cin,kh,kw)
    Cin = W.shape[1]
    std = torch.tensor(STD).view(1, -1, 1, 1)
    if Cin == 3:
        Weff = W / std
    else:
        Weff = W / MIN_STD  # patch-embed sometimes != 3 in-ch; conservative
    rowsum = Weff.abs().sum(dim=(1, 2, 3))  # (Cout,)
    return float(rowsum.max())


@torch.no_grad()
def emax_for(model, hook, loader, n):
    out = []
    seen = 0
    for batch in loader:
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        imgs = imgs.to(DEVICE)
        model(imgs)
        out.append(hook.features.flatten(1).float())  # (B, C*H*W) post-conv
        seen += imgs.shape[0]
        if seen >= n:
            break
    return torch.cat(out, 0)[:n]


for m in MODELS:
    arch, wp = config.MODELS[m]
    model = load_normalized_model(arch, wp, num_classes=config.NUM_CLASSES, device=DEVICE)
    model.eval()
    _, layer = find_first_conv_in_normalized(model)
    W1 = w1_norm(layer)
    bound = W1 * EPS_NORM
    idl = get_id_loader("cifar100", batch_size=64, num_workers=4, train=False)
    with FirstLayerHook(model) as hook:
        clean = emax_for(model, hook, idl, N)
        for atk in ATTACKS:
            h5 = config.ADV_DIR / f"{m}_{atk}.h5"
            if not h5.exists():
                print(f"{m},{atk},MISSING")
                continue
            al = adv_loader_from_h5(h5, batch_size=64, num_workers=4)
            adv = emax_for(model, hook, al, N)
            al.dataset.close()
            k = min(clean.shape[0], adv.shape[0])
            e = (adv[:k] - clean[:k]).abs()
            emax = e.amax(dim=1).cpu().numpy()  # per-sample ||e_m||_inf
            tau = emax / bound
            print(f"{m},{atk},W1={W1:.4f},eps_norm={EPS_NORM:.4f},bound={bound:.4f},"
                  f"emax_mean={emax.mean():.4f},emax_p95={np.percentile(emax,95):.4f},"
                  f"emax_max={emax.max():.4f},"
                  f"tau_mean={tau.mean():.4f},tau_p95={np.percentile(tau,95):.4f},"
                  f"tau_max={tau.max():.4f},n={k}")
    del model
    torch.cuda.empty_cache()
