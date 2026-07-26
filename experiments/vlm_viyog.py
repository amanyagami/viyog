"""VLM track — Viyog on Qwen2-VL-2B vision encoder (feature-collision attack).

Extends Viyog to a VLM: hook the vision tower's first conv (patch_embed.proj,
Conv3d) and test whether the first-layer activation-norm separates ADV from OOD,
exactly as for the classifiers — but ADV here is a *vision-encoder feature-collision*
PGD (maximize L2 deviation of the pooled patch embedding under an L∞ pixel budget),
since a VLM has no class logits.

Run a smoke test first (verifies forward + hook + attack gradient):
    CUDA_VISIBLE_DEVICES=4 python experiments/vlm_viyog.py --smoke 4
Full run (ID + OOD + ADV inf-norms -> ADV-vs-OOD AUROC):
    CUDA_VISIBLE_DEVICES=4 python experiments/vlm_viyog.py --n 1000
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("HF_HOME", "/mnt/data1/asing725/viyog/weights/.hf")
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
from sklearn.metrics import roc_auc_score

DEV = "cuda:0"
MID = "Qwen/Qwen2-VL-2B-Instruct"
EPS = 8 / 255


def load_vlm():
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MID, torch_dtype=torch.float32, device_map=DEV).eval()
    proc = AutoProcessor.from_pretrained(MID, min_pixels=224 * 224, max_pixels=224 * 224)
    return model, proc


class PatchEmbedHook:
    """Capture the Conv3d patch-embed output (the Viyog first-layer activation)."""
    def __init__(self, visual):
        self.feat = None
        # first conv in the vision tower
        conv = next(m for _, m in visual.named_modules()
                    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)))
        self.h = conv.register_forward_hook(lambda mod, i, o: setattr(self, "feat", o))

    def close(self):
        self.h.remove()


def vision_embed(model, pixel_values, grid_thw):
    """Run the vision tower → (num_patches, dim) embedding tensor (graph retained)."""
    out = model.model.visual(pixel_values, grid_thw=grid_thw)
    if torch.is_tensor(out):
        return out
    t = getattr(out, "last_hidden_state", None)
    return t if t is not None else out.pooler_output


EPS_PV = 0.12   # L∞ budget in the (normalized) vision-encoder input space (~8/255 image)


def feature_collision_attack(model, pv, thw, steps):
    """Random-start PGD maximizing L2 deviation of the vision embedding.
    Random start is essential: at the clean point the squared-deviation
    gradient is exactly zero (emb == emb_clean)."""
    with torch.no_grad():
        emb_clean = vision_embed(model, pv, thw).detach()
    delta = (torch.rand_like(pv) * 2 - 1) * EPS_PV       # random start in the ball
    alpha = 2.5 * EPS_PV / steps
    for _ in range(steps):
        delta.requires_grad_(True)
        emb = vision_embed(model, pv + delta, thw)
        loss = -(emb - emb_clean).pow(2).mean()           # maximize deviation
        g = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            delta = (delta - alpha * g.sign()).clamp_(-EPS_PV, EPS_PV)
    return (pv + delta).detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="run N-image smoke test and exit")
    ap.add_argument("--n", type=int, default=1000, help="images per split for the full run")
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    print("loading Qwen2-VL-2B ...", flush=True)
    model, proc = load_vlm()
    for p in model.parameters():
        p.requires_grad_(False)
    hook = PatchEmbedHook(model.model.visual)
    print("loaded; patch-embed hook attached", flush=True)

    from PIL import Image
    cifar = tv.datasets.CIFAR100("/mnt/data1/asing725/viyog/data/cifar", train=False, download=True)

    def to_pil(i):
        return Image.fromarray(cifar.data[i]).convert("RGB")

    @torch.no_grad()
    def proc_one(pil):
        out = proc.image_processor(images=[pil], return_tensors="pt")
        return out["pixel_values"].to(DEV), out["image_grid_thw"].to(DEV)

    def inf_norm_from_feat():
        f = hook.feat  # (num_patches, dim) or (dim, num_patches) depending on conv
        return f.detach().abs().max().item()

    if args.smoke:
        print(f"=== SMOKE TEST ({args.smoke} imgs) ===", flush=True)
        pil = to_pil(0)
        pv, thw = proc_one(pil)
        print(f"  pixel_values {tuple(pv.shape)}  grid_thw {thw.tolist()}", flush=True)
        emb0 = vision_embed(model, pv, thw)
        print(f"  vision embed {tuple(emb0.shape)}  patch-embed act {tuple(hook.feat.shape)}", flush=True)
        clean_norm = inf_norm_from_feat()
        pv_adv = feature_collision_attack(model, pv, thw, args.steps)
        with torch.no_grad():
            emb_adv = vision_embed(model, pv_adv, thw)
        adv_norm = inf_norm_from_feat()
        dev = (emb_adv - emb0.detach()).pow(2).mean().sqrt().item()
        print(f"  clean inf-norm={clean_norm:.4f}  adv inf-norm={adv_norm:.4f}  feat-dev(L2)={dev:.4f}", flush=True)
        print("  SMOKE OK — forward + hook + attack gradient all work", flush=True)
        hook.close()
        return

    # ---- full run: collect patch-embed inf-norms for ID, OOD, ADV ----
    def id_norms(n):
        out = []
        for i in range(n):
            pv, thw = proc_one(to_pil(i))
            with torch.no_grad():
                vision_embed(model, pv, thw)
            out.append(inf_norm_from_feat())
        return np.array(out)

    def adv_norms(n):
        out = []
        for i in range(n):
            pv, thw = proc_one(to_pil(i))
            pv_adv = feature_collision_attack(model, pv, thw, args.steps)
            with torch.no_grad():
                vision_embed(model, pv_adv, thw)
            out.append(inf_norm_from_feat())
        return np.array(out)

    print(f"=== FULL: {args.n} imgs/split ===", flush=True)
    idn = id_norms(args.n)
    advn = adv_norms(args.n)
    # OOD: SVHN test images
    svhn = tv.datasets.SVHN("/mnt/data1/asing725/viyog/data/ood/svhn", split="test", download=True)
    from PIL import Image as I2
    oodn = []
    for i in range(args.n):
        pv, thw = proc_one(I2.fromarray(np.transpose(svhn.data[i], (1, 2, 0))).convert("RGB"))
        with torch.no_grad():
            vision_embed(model, pv, thw)
        oodn.append(inf_norm_from_feat())
    oodn = np.array(oodn)
    # ADV vs OOD AUROC (directionless)
    y = np.r_[np.ones(len(advn)), np.zeros(len(oodn))]
    s = np.r_[advn, oodn]
    auroc = max(roc_auc_score(y, s), roc_auc_score(y, -s))
    print(f"  ID inf-norm μ={idn.mean():.3f} | ADV μ={advn.mean():.3f} | OOD μ={oodn.mean():.3f}", flush=True)
    print(f"  ADV-vs-OOD AUROC (patch-embed inf-norm) = {auroc:.4f}", flush=True)
    np.savez("/mnt/data1/asing725/viyog/results/analysis/vlm_qwen2vl_viyog.npz",
             id=idn, adv=advn, ood=oodn, auroc=auroc)
    hook.close()


if __name__ == "__main__":
    main()
