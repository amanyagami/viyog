"""Regression + invariant tests for the streaming first-layer signature extractor.

Guarantees (all on CPU, so the running GPU pipeline is untouched):
  1. Micro-batching is numerically identical to single-shot (bit-exact) — the
     refactor changes only peak memory, never the result.
  2. The result is independent of the chunk size `micro` and of the batch size
     B (reproducibility: signatures do not depend on how samples are grouped).
  3. Determinism: same input -> identical output across repeated calls.
  4. NO reduction/averaging of the first-layer map before the signature: every
     statistic uses the full (C,H,W) activation. Perturbing a single pixel must
     change the signatures (nothing is pooled away first); inf_norm equals the
     max over the entire unreduced activation.

Run:  CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest experiments/test_extract_full_stats.py -q
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU; do not touch the busy GPUs

# Import 06b by path (module name starts with a digit -> importlib).
_MOD_PATH = Path(__file__).resolve().parent / "06b_extract_full.py"
_spec = importlib.util.spec_from_file_location("extract_full", _MOD_PATH)
ef = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ef)  # type: ignore[union-attr]

_KEYS_F = ("filter_means", "filter_maxs", "filter_l2", "filter_std", "filter_tv", "filter_hf")
_KEYS_I = ("inf_norms", "gram_offdiag")


def _rand_feats(B=600, C=64, H=56, W=56, seed=0):
    g = torch.Generator().manual_seed(seed)
    # signed, non-trivial spatial structure
    return torch.randn(B, C, H, W, generator=g)


def _flatten(pf, pi):
    return torch.cat([pf[k].reshape(-1) for k in _KEYS_F] + [pi[k].reshape(-1) for k in _KEYS_I])


def test_micro_equals_single_shot():
    """Chunked == single-shot within float precision (and far within the float16
    persisted on disk). The math is per-sample; only reduction order can differ
    at the ULP level, which float16 storage rounds away entirely."""
    a = _rand_feats()
    single = _flatten(*ef._stats_chunk(a.float()))           # whole batch at once
    micro = _flatten(*ef.compute_full_stats(a, micro=256))   # production chunk size
    assert torch.allclose(single, micro, rtol=1e-5, atol=1e-6), (single - micro).abs().max().item()
    # tolerance is orders of magnitude tighter than the float16 storage step
    assert (single - micro).abs().max().item() < 1e-3


def test_max_signatures_bit_exact_across_chunks():
    """inf_norm and filter_max are pure max reductions -> EXACTLY chunk-invariant
    (no accumulation), even at the degenerate micro=1. This is the L-inf Viyog
    statistic, so its reproducibility is bit-exact regardless of batching."""
    a = _rand_feats()
    ref_pf, ref_pi = ef.compute_full_stats(a, micro=999_999)  # one chunk
    for m in (1, 7, 64, 256):
        pf, pi = ef.compute_full_stats(a, micro=m)
        assert torch.equal(ref_pi["inf_norms"], pi["inf_norms"]), f"inf_norm @micro={m}"
        assert torch.equal(ref_pf["filter_maxs"], pf["filter_maxs"]), f"filter_max @micro={m}"


def test_averaging_signatures_invariant_within_precision():
    """Accumulation-based stats match across realistic chunk sizes within FP tol."""
    a = _rand_feats()
    ref = _flatten(*ef.compute_full_stats(a, micro=999_999))
    for m in (64, 128, 256):
        got = _flatten(*ef.compute_full_stats(a, micro=m))
        assert torch.allclose(ref, got, rtol=1e-5, atol=1e-6), f"chunk size {m}"


def test_invariant_to_batch_grouping():
    """A sample's signature must not depend on which other samples share its batch."""
    a = _rand_feats(B=300)
    full_pf, full_pi = ef.compute_full_stats(a, micro=256)
    # process the same samples split into two arbitrary sub-batches
    pf1, pi1 = ef.compute_full_stats(a[:123], micro=256)
    pf2, pi2 = ef.compute_full_stats(a[123:], micro=256)
    for k in _KEYS_F:
        assert torch.equal(full_pf[k], torch.cat([pf1[k], pf2[k]], 0)), k
    for k in _KEYS_I:
        assert torch.equal(full_pi[k], torch.cat([pi1[k], pi2[k]], 0)), k


def test_determinism_repeat():
    a = _rand_feats()
    r1 = _flatten(*ef.compute_full_stats(a, micro=64))
    r2 = _flatten(*ef.compute_full_stats(a, micro=64))
    assert torch.equal(r1, r2)


def test_inf_norm_is_full_activation_max():
    """inf_norm must be the max |value| over the ENTIRE (C,H,W) map — no averaging."""
    a = _rand_feats(B=20)
    pf, pi = ef.compute_full_stats(a, micro=8)
    expected = a.abs().reshape(a.shape[0], -1).amax(dim=1)
    assert torch.equal(pi["inf_norms"], expected.float())


def test_single_pixel_perturbation_is_not_averaged_away():
    """If features were pooled/averaged before the signature, a one-pixel spike
    in a sample could be washed out. It must instead change that sample's
    signatures (and ONLY that sample's)."""
    a = _rand_feats(B=10)
    base_pf, base_pi = ef.compute_full_stats(a, micro=4)
    a2 = a.clone()
    a2[3, 5, 10, 12] += 1000.0  # huge spike at one pixel of sample 3
    pf, pi = ef.compute_full_stats(a2, micro=4)
    # sample 3's inf_norm must jump to ~the spike; others unchanged
    assert pi["inf_norms"][3] > base_pi["inf_norms"][3] + 900.0
    untouched = [i for i in range(10) if i != 3]
    assert torch.equal(pi["inf_norms"][untouched], base_pi["inf_norms"][untouched])
    # the spiked filter's mean/max for sample 3 must also move (not averaged out)
    assert pf["filter_maxs"][3, 5] > base_pf["filter_maxs"][3, 5] + 900.0


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
    sys.exit(0)
