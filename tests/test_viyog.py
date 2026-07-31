"""Tests for the dormant-band roughness detector (paper method)."""

import gc
import threading
import weakref

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from viyog import Viyog


class _SmallConvNet(torch.nn.Module):
    """Tiny model with a ``conv1`` attribute so Viyog discovers it predictably."""

    def __init__(self, in_ch: int = 3, out_ch: int = 8) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(out_ch, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def _smooth_images(n: int, size: int = 16) -> torch.Tensor:
    """Low-frequency (spatially smooth) images: upsampled coarse noise."""
    coarse = torch.randn(n, 3, size // 4, size // 4)
    return torch.nn.functional.interpolate(coarse, size=size, mode="bilinear", align_corners=False)


def _loader(x: torch.Tensor, bs: int = 4) -> DataLoader:
    return DataLoader(TensorDataset(x, torch.zeros(len(x), dtype=torch.long)), batch_size=bs)


def test_fit_sets_fitted_state() -> None:
    torch.manual_seed(0)
    v = Viyog(_SmallConvNet(), device="cpu")
    v.fit(_loader(_smooth_images(16)))
    assert v.n_channels_ == 8
    assert v.dorm_idx_ is not None and v.dorm_idx_.numel() == max(1, int(8 * 0.10))
    assert v.id_profile_ is not None and v.id_profile_.numel() == 8
    assert isinstance(v.id_score_mean_, float)
    v.close()


def test_score_before_fit_raises() -> None:
    v = Viyog(_SmallConvNet(), device="cpu")
    with pytest.raises(RuntimeError):
        v.score(torch.randn(2, 3, 16, 16))
    v.close()


def test_no_conv_layer_raises() -> None:
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 16 * 16, 2))
    with pytest.raises(RuntimeError):
        Viyog(model, device="cpu")


def test_score_shapes_batch_and_loader() -> None:
    torch.manual_seed(0)
    with Viyog(_SmallConvNet(), device="cpu") as v:
        v.fit(_loader(_smooth_images(16)))
        batch = v.score(torch.randn(5, 3, 16, 16))
        assert batch.shape == (5,)
        whole = v.score_loader(_loader(_smooth_images(12)))
        assert whole.shape == (12,)


def test_hf_perturbation_scores_higher() -> None:
    """Core mechanism: injecting high-frequency residue raises the roughness score.

    Adversarial perturbations add broadband high-frequency noise to the input; the
    dormant-band total variation should rise. We fit on smooth ID images, then
    compare smooth vs HF-perturbed versions of the same images.
    """
    torch.manual_seed(0)
    ref = _smooth_images(64)
    hf = ref + 0.15 * torch.randn_like(ref)  # broadband high-frequency residue
    with Viyog(_SmallConvNet(), device="cpu") as v:
        v.fit(_loader(ref))
        smooth_scores = v.score(ref)
        rough_scores = v.score(hf)
    assert rough_scores.mean() > smooth_scores.mean()


def test_dead_channels_excluded_from_dorm_band() -> None:
    """A permanently-dead channel (zero weight+bias) must never enter the band."""
    torch.manual_seed(0)
    model = _SmallConvNet()
    with torch.no_grad():
        model.conv1.weight[0].zero_()
        model.conv1.bias[0].zero_()  # channel 0 output is always 0 -> profile 0
    with Viyog(model, device="cpu", dorm_pct=0.5) as v:
        v.fit(_loader(_smooth_images(16)))
        assert 0 not in v.dorm_idx_.tolist()


def test_explicit_layer_and_dorm_pct() -> None:
    torch.manual_seed(0)
    model = _SmallConvNet(out_ch=20)
    with Viyog(model, device="cpu", layer="conv1", dorm_pct=0.25) as v:
        assert v.layer_name_ == "conv1"
        v.fit(_loader(_smooth_images(16)))
        assert v.dorm_idx_.numel() == int(20 * 0.25)


def test_conv1d_signal_model() -> None:
    """1-D signal path: a Conv1d first layer yields (B, C, L) maps."""

    class _Sig(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = torch.nn.Conv1d(1, 6, 5, padding=2)
            self.fc = torch.nn.Linear(6, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(torch.relu(self.conv1(x)).mean(-1))

    torch.manual_seed(0)
    x = torch.randn(10, 1, 32)
    with Viyog(_Sig(), device="cpu", dorm_pct=0.5) as v:
        v.fit(_loader(x))
        scores = v.score(x)
        assert scores.shape == (10,)


def test_bounded_score_is_monotone_and_in_unit_interval() -> None:
    raw = torch.tensor([-3.0, -1.0, 0.0, 2.0, 5.0])
    b = Viyog.bounded_score(raw)
    assert torch.all((b > 0) & (b < 1))
    assert torch.all(b[1:] > b[:-1])  # strictly increasing


def test_center_shifts_but_preserves_order() -> None:
    torch.manual_seed(0)
    x = _smooth_images(16)
    with Viyog(_SmallConvNet(), device="cpu") as v:
        v.fit(_loader(x))
        raw = v.score(x)
        cen = v.score(x, center=True)
    assert torch.allclose(raw - v.id_score_mean_, cen, atol=1e-5)


class _HeadlessConvNet(torch.nn.Module):
    """No classification head, so its forward is unbatched-safe end to end --
    isolates Viyog's own batching logic from an unrelated Linear-head shape
    mismatch on unbatched input (which is a property of the wrapped model,
    not of Viyog)."""

    def __init__(self, in_ch: int = 3, out_ch: int = 8) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.conv1(x))


def test_unbatched_input_matches_batch_of_one() -> None:
    """A single (C, H, W) image -- no batch dim -- must score identically to
    the same image with an explicit batch dim of 1, not be silently
    misread as a (B, C, L) 1-D-signal batch (regression: a Conv2d hook's
    unbatched 3-D output was previously misdispatched to the Conv1d branch,
    returning a plausible but meaningless score with no error)."""
    torch.manual_seed(0)
    x = _smooth_images(16)
    with Viyog(_HeadlessConvNet(), device="cpu") as v:
        v.fit(_loader(x))
        image = x[0]
        assert image.dim() == 3  # (C, H, W), genuinely unbatched
        unbatched = v.score(image)
        batched = v.score(image.unsqueeze(0))
    assert unbatched.shape == batched.shape == (1,)
    assert torch.allclose(unbatched, batched, atol=1e-6)


def test_conv1d_batch_still_uses_1d_branch() -> None:
    """A genuine (B, C, L) Conv1d batch must not be affected by the unbatched-
    Conv2d disambiguation above -- same dim() as an unbatched Conv2d map, but
    a different, legitimate meaning."""

    class _Sig(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = torch.nn.Conv1d(1, 6, 5, padding=2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.conv1(x))

    torch.manual_seed(0)
    x = torch.randn(5, 1, 32)
    with Viyog(_Sig(), device="cpu", dorm_pct=0.5) as v:
        v.fit(_loader(x))
        scores = v.score(x)
    assert scores.shape == (5,)  # one score per signal in the batch, not per-length


def test_instance_freed_without_close() -> None:
    """The forward hook must hold a weak reference to the detector: a strong
    reference would create model -> hooks -> closure -> self cycle, so a
    Viyog dropped without close()/context-manager would survive until the
    next cyclic-GC pass instead of being freed immediately (regression)."""
    torch.manual_seed(0)
    model = _HeadlessConvNet()
    gc.disable()
    try:
        v = Viyog(model, device="cpu")
        v.fit(_loader(_smooth_images(16)))
        ref = weakref.ref(v)
        del v
        assert ref() is None, "instance survived del -- hook holds a strong reference"
        assert len(model._forward_hooks) == 0, "hook was not removed with the instance"
        model(_smooth_images(2))  # must not raise with the referent gone
    finally:
        gc.enable()


def test_concurrent_score_is_thread_safe() -> None:
    """Two threads calling score() on the SAME detector instance must not
    observe each other's activations (regression: the hook wrote into shared
    instance state with no locking around the capture-forward-read
    sequence)."""
    torch.manual_seed(0)
    with Viyog(_HeadlessConvNet(), device="cpu") as v:
        v.fit(_loader(_smooth_images(16)))
        a = torch.zeros(6, 3, 16, 16)
        b = torch.ones(6, 3, 16, 16) * 0.7
        ref_a, ref_b = v.score(a), v.score(b)  # serial ground truth

        out: dict[str, torch.Tensor] = {}

        def run(key: str, x: torch.Tensor) -> None:
            out[key] = v.score(x)

        pairs = [("a", a), ("b", b)] * 8
        threads = [threading.Thread(target=run, args=(f"{k}{i}", x))
                   for i, (k, x) in enumerate(pairs)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    for key, result in out.items():
        ref = ref_a if key.startswith("a") else ref_b
        assert torch.allclose(result, ref, atol=1e-6), f"{key} diverged from serial reference"
