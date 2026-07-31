"""Step 6b – One-pass RICH first-layer feature extraction.

Single forward pass per batch. While the full (B, C, H, W) first-layer map is
in VRAM, compute the ENTIRE statistic battery any signature could need — the
legacy per-filter reductions (mean/max/RMS) PLUS spatial/frequency descriptors
(std, total-variation, high-frequency energy ratio) and a cross-filter
co-activation scalar (Gram off-diagonal energy fraction). Only the compact
per-filter summaries are written; the (B,C,H,W) tensor is freed before the next
batch, so memory use is bounded by one batch regardless of dataset size.

Large feature batch sizes (config.MODEL_FEATURE_BATCH) keep the GPU busy;
torch.no_grad() means activations are freed immediately after the hook fires.

Outputs (skip-if-exists, resumable):
    <features>/featfull_<model>_id.h5
    <features>/featfull_<model>_ood_<name>.h5     (10 OOD sets)
    <features>/featfull_<model>_adv_<attack>.h5

Reproducible: deterministic loaders (shuffle off), fixed sample cap, fixed
EuroSAT split seed, cudnn deterministic. Runs across all models/datasets/OOD
via the same --dataset/--models/--attacks flags as step 06.

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/06b_extract_full.py --dataset cifar10
    CUDA_VISIBLE_DEVICES=1 python experiments/06b_extract_full.py --dataset cifar100 \
        --models vit_base --attacks fgsm bim pgd apgd_ce
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (
    ATTACKS,
    DEVICE,
    MODEL_FEATURE_BATCH,
    N_FEATURE_SAMPLES,
)
from data_utils import (
    FullFeatureH5Writer,
    adv_loader_from_h5,
    get_id_loader,
    get_ood_loader,
)
from model_utils import FirstLayerHook, find_first_conv_in_normalized, load_normalized_model
from tqdm import tqdm

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
_EPS = 1e-8


def _seed() -> None:
    """Pin every RNG / algorithm knob so extraction is bit-reproducible.

    cuBLAS needs a fixed workspace for deterministic GEMMs; this must be set
    before the first CUDA matmul, which `_seed()` (called at the top of main)
    precedes. `warn_only=True` keeps a backbone with a non-deterministic op from
    crashing — the first-conv feature path (conv + normalize) is deterministic
    under cudnn.deterministic, which is what the signatures depend on.
    """
    import numpy as np

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-pass rich first-layer feature extraction")
    p.add_argument("--dataset", default="cifar100", choices=list(config.DATASET_SPECS))
    p.add_argument("--models", nargs="+", default=list(config.MODEL_ARCHS),
                   choices=list(config.MODEL_ARCHS), metavar="MODEL")
    p.add_argument("--attacks", nargs="+", default=list(ATTACKS),
                   choices=list(ATTACKS), metavar="ATTACK",
                   help="Adversarial split(s) to extract (default: all available)")
    p.add_argument("--cleanup-adv", action="store_true",
                   help="Delete each adversarial source h5 once its signatures are "
                        "extracted (the raw images are consumed) — bounds disk use for "
                        "multi-dataset runs. The compact featfull file is what's kept.")
    p.add_argument("--batch", type=int, default=None,
                   help="Override the per-model feature batch size (default: "
                        "config.MODEL_FEATURE_BATCH, tuned for large-VRAM GPUs). "
                        "Lower this on a smaller card, e.g. --batch 128 for 8-16 GB. "
                        "Correctness is unaffected; only throughput changes.")
    p.add_argument("--ood", nargs="+", default=None, choices=list(config.OOD_DATASETS),
                   metavar="OOD",
                   help="Restrict the OOD universe (default: all 10). First-run "
                        "OOD downloads dominate this step's wall-clock, so a subset "
                        "such as '--ood cifar10 svhn' gives a fast end-to-end check. "
                        "NOTE: T3 (OOD-vs-ADV) is then averaged over only those sets "
                        "and is NOT comparable to the paper's 10-set T3.")
    return p.parse_args()


# Row-chunk size for the statistic computation. The forward pass still uses the
# large config.MODEL_FEATURE_BATCH (GPU stays busy), but the heavy (B,C,H,W)
# stat intermediates (abs map, avg-pool blur, Gram) are computed in chunks of
# this many samples so peak memory is bounded by one chunk rather than the whole
# feature batch. Per-sample statistics are independent, so the chunked result is
# numerically identical to a single-shot computation.
_STAT_MICRO = 256


@torch.no_grad()
def _stats_chunk(a: torch.Tensor):
    """Per-filter + per-image statistics for one (m,C,H,W) activation chunk.

    Args:
        a: First-layer activations for ``m`` samples; must already be float and
            4-D. Every intermediate is scoped to this call, so the chunk's
            (m,C,H,W) tensors are freed as soon as it returns.

    Returns:
        ``(per_filter, per_image)`` — dicts of ``(m,C)`` and ``(m,)`` tensors.
    """
    B, C, H, W = a.shape
    absa = a.abs()
    HW = H * W

    mean = absa.mean(dim=(2, 3))                              # (m,C)
    mx = absa.amax(dim=(2, 3))
    l2 = a.pow(2).mean(dim=(2, 3)).clamp_min(0).sqrt()
    std = a.std(dim=(2, 3))
    infn = absa.flatten(1).amax(dim=1)                       # (m,)

    # Total variation (spatial roughness), normalised → shape, not magnitude.
    if H > 1 and W > 1:
        dh = (a[:, :, 1:, :] - a[:, :, :-1, :]).abs().mean(dim=(2, 3))
        dw = (a[:, :, :, 1:] - a[:, :, :, :-1]).abs().mean(dim=(2, 3))
        tv = (dh + dw) / (mean + 1e-6)
    else:
        tv = torch.zeros_like(mean)

    # High-frequency energy ratio via a local-mean high-pass (avg-pool blur).
    if H >= 3 and W >= 3:
        blur = F.avg_pool2d(a, kernel_size=3, stride=1, padding=1)
        hf = a - blur
        hf_ratio = hf.pow(2).mean(dim=(2, 3)) / (a.pow(2).mean(dim=(2, 3)) + _EPS)
    else:
        hf_ratio = torch.zeros_like(mean)

    # Cross-filter co-activation: off-diagonal Gram energy fraction.
    # ||A Aᵀ||_F = ||Aᵀ A||_F, so build the Gram on whichever dim is smaller.
    A = a.flatten(2)                                         # (m,C,HW)
    if C <= HW:
        M = torch.bmm(A, A.transpose(1, 2))                 # (m,C,C)
    else:
        M = torch.bmm(A.transpose(1, 2), A)                 # (m,HW,HW), same ‖·‖_F
    fro2 = M.pow(2).sum(dim=(1, 2))
    diag2 = (l2.pow(2) * HW).pow(2).sum(dim=1)              # Σ_c (Σ_s a_cs²)²
    gram_off = (1.0 - diag2 / (fro2 + _EPS)).clamp(0.0, 1.0)

    per_filter = {"filter_means": mean, "filter_maxs": mx, "filter_l2": l2,
                  "filter_std": std, "filter_tv": tv, "filter_hf": hf_ratio}
    per_image = {"inf_norms": infn, "gram_offdiag": gram_off}
    return per_filter, per_image


@torch.no_grad()
def compute_full_stats(feats: torch.Tensor, micro: int = _STAT_MICRO):
    """Stream the full statistic battery over one (B,C,H,W) activation map.

    The signatures are computed in row-chunks of ``micro`` samples; the large
    feature-derived intermediates never exceed one chunk and are freed between
    chunks, while only the compact (B,C) / (B,) signature tensors accumulate.
    Because every statistic is per-sample, the chunked result is numerically
    identical to processing the whole batch at once.

    Args:
        feats: First-layer activations ``(B,C,H,W)`` for the current batch. The
            caller discards ``feats`` afterward; nothing is retained here.
        micro: Maximum samples processed per chunk.

    Returns:
        ``(per_filter, per_image)`` — dicts of ``(B,C)`` and ``(B,)`` tensors.
    """
    a_full = feats
    if a_full.ndim != 4:                  # safety: treat (B,C) as 1×1 maps
        a_full = a_full.unsqueeze(-1).unsqueeze(-1)
    B = a_full.shape[0]
    if B <= micro:
        return _stats_chunk(a_full.float())

    pf_acc: dict[str, list[torch.Tensor]] = {}
    pi_acc: dict[str, list[torch.Tensor]] = {}
    for s in range(0, B, micro):
        pf, pi = _stats_chunk(a_full[s:s + micro].float())
        for k, v in pf.items():
            pf_acc.setdefault(k, []).append(v)
        for k, v in pi.items():
            pi_acc.setdefault(k, []).append(v)
        del pf, pi
    per_filter = {k: torch.cat(v, dim=0) for k, v in pf_acc.items()}
    per_image = {k: torch.cat(v, dim=0) for k, v in pi_acc.items()}
    return per_filter, per_image


@torch.no_grad()
def extract(norm_model, loader, out_path: Path, n_total: int, n_filters: int,
            n_classes: int, desc: str) -> None:
    if out_path.exists():
        print(f"  [skip] {out_path.name}")
        return
    norm_model.eval()
    with (
        FirstLayerHook(norm_model) as hook,
        FullFeatureH5Writer(out_path, n_total, n_filters, n_classes=n_classes) as writer,
    ):
        written = 0
        for batch in tqdm(loader, desc=f"  {desc}", dynamic_ncols=True, leave=False):
            if written >= n_total:
                break
            if isinstance(batch, (list, tuple)):
                imgs = batch[0]
                labels = batch[1] if len(batch) > 1 else torch.full((imgs.shape[0],), -1)
            else:
                imgs, labels = batch, torch.full((batch.shape[0],), -1)

            imgs = imgs.to(DEVICE, non_blocking=True)
            logits = norm_model(imgs)            # one forward pass; triggers hook
            feats = hook.features                # (B,C,H,W)
            per_filter, per_image = compute_full_stats(feats)

            take = n_total - written
            sl = slice(0, take)
            writer.write_batch(
                {k: v[sl] for k, v in per_filter.items()},
                {k: v[sl] for k, v in per_image.items()},
                labels[:take], logits[sl],
            )
            written += min(take, logits.shape[0])
            del feats, per_filter, per_image, logits   # free before next batch
    print(f"  ✓ {out_path.name}  [{written} samples]")


def process_model(model_name: str, arch: str, weight_path, attacks: list[str],
                  dataset: str, cleanup_adv: bool = False,
                  batch_override: int | None = None,
                  ood_subset: list[str] | None = None) -> None:
    feat_batch = batch_override if batch_override is not None else MODEL_FEATURE_BATCH[model_name]
    n_classes = config.NUM_CLASSES
    norm_model = load_normalized_model(arch, weight_path, num_classes=n_classes, device=DEVICE)
    _, layer = find_first_conv_in_normalized(norm_model)
    n_filters = layer.out_channels
    print(f"\n  === {model_name}  (batch={feat_batch}, filters={n_filters}) ===")
    fdir = config.FEATURES_DIR

    # ID
    idl = get_id_loader(dataset, batch_size=feat_batch, num_workers=4, train=False)
    extract(norm_model, idl, fdir / f"featfull_{model_name}_id.h5",
            min(N_FEATURE_SAMPLES, len(idl.dataset)), n_filters, n_classes,
            f"{model_name}/id")

    # OOD (10-set universe minus self)
    #
    # A failed OOD split silently shrinks the universe T3 (OOD-vs-ADV) averages
    # over, so the reported number would no longer mean what the paper's means.
    # Only *environmental* failures (an unreachable dataset mirror -- GTSRB's
    # has intermittent outages) are survivable: warn, keep going, and report the
    # shortfall loudly at the end. Anything else -- above all a CUDA OOM, the
    # likely failure on a smaller or shared GPU -- is a configuration problem
    # that must stop the run rather than quietly degrade the result.
    failed_ood: list[str] = []
    for ood in (ood_subset if ood_subset is not None else config.OOD_DATASETS):
        try:
            ol = get_ood_loader(ood, batch_size=feat_batch, num_workers=4,
                                max_samples=N_FEATURE_SAMPLES)
            extract(norm_model, ol, fdir / f"featfull_{model_name}_ood_{ood}.h5",
                    min(N_FEATURE_SAMPLES, len(ol.dataset)), n_filters, n_classes,
                    f"{model_name}/ood-{ood}")
        except torch.cuda.OutOfMemoryError:
            print(f"\n  [FATAL] CUDA out of memory on OOD split '{ood}' "
                  f"(feature batch = {feat_batch}).")
            print( "          Re-run with a smaller batch, e.g.  --batch 64")
            print( "          (or use reproduce_t1.sh, which sizes the batch to your GPU).")
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except (OSError, RuntimeError, ValueError) as e:
            # Download/mirror/decode failures: survivable, but never silent.
            print(f"  [warn] OOD {ood} failed ({type(e).__name__}): {e}")
            failed_ood.append(ood)

    if failed_ood:
        print(f"\n  [!] {len(failed_ood)} OOD split(s) missing: {', '.join(failed_ood)}")
        print( "      T2 (ID-vs-ADV) is unaffected. T3 (OOD-vs-ADV) will be averaged")
        print( "      over fewer OOD sets than the paper and is NOT directly comparable.")
        print( "      Re-run this script to retry only the missing splits.")

    # ADV
    for atk in attacks:
        h5 = config.ADV_DIR / f"{model_name}_{atk}.h5"
        if not h5.exists():
            print(f"  [skip] {h5.name} not found")
            continue
        al = adv_loader_from_h5(h5, batch_size=feat_batch, num_workers=4)
        out = fdir / f"featfull_{model_name}_adv_{atk}.h5"
        extract(norm_model, al, out,
                min(N_FEATURE_SAMPLES, len(al.dataset)), n_filters, n_classes,
                f"{model_name}/adv-{atk}")
        al.dataset.close()
        # The raw adversarial images are consumed once signatures are extracted;
        # drop them to bound disk use (only the compact featfull file is kept).
        if cleanup_adv and out.exists() and out.stat().st_size > 10_000:
            try:
                h5.unlink()
                print(f"  [cleanup] removed {h5.name} (signatures extracted)")
            except OSError as e:  # noqa: BLE001
                print(f"  [cleanup] could not remove {h5.name}: {e}")

    del norm_model
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    args = _parse_args()
    _seed()
    config.set_dataset(args.dataset)
    print(f"=== Step 6b: rich one-pass feature extraction [{args.dataset}] ===")
    print(f"  models: {args.models}  attacks: {args.attacks}")
    _ood = args.ood if args.ood is not None else list(config.OOD_DATASETS)
    print(f"  OOD ({len(_ood)}): {_ood}")
    if args.ood is not None:
        print("  [!] OOD universe restricted -- T3 is NOT comparable to the paper's 10-set T3.")
    print(f"  output: {config.FEATURES_DIR}")
    for m in args.models:
        arch, wp = config.MODELS[m]
        process_model(m, arch, wp, args.attacks, args.dataset,
                      cleanup_adv=args.cleanup_adv, batch_override=args.batch,
                      ood_subset=args.ood)
    files = sorted(config.FEATURES_DIR.glob("featfull_*.h5"))
    print(f"\n  {len(files)} rich feature files in {config.FEATURES_DIR}")
    print("=== Step 6b complete ===")


if __name__ == "__main__":
    main()
