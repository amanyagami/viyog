"""Viyog — separating adversarial (ADV) from out-of-distribution (OOD) inputs.

Viyog is a training-free, gradient-free, post-hoc detector. It attaches a forward
hook to a model's **first convolutional layer** and reads the *dormant-band
roughness* of that layer's activation map: a magnitude-normalised **total
variation** (average absolute change between neighbouring pixels) averaged over the
channels that stay quietest on in-distribution (ID) data.

The mechanism: gradient-based adversarial attacks inject broadband, high-frequency
residue into the otherwise-silent first-layer channels, making them spatially
*jagged*; natural inputs — both ID and OOD — leave those channels *smooth*. So the
roughness statistic ``V(x)`` is high for ADV inputs and low for everything else,
which lets a downstream stage tell the two anomaly regimes apart. The detector adds
no parameters, never touches the backward pass, and stores only ``O(C)`` bytes of
state (the dormant-channel ranking + one ID mean).

Recommended usage::

    from viyog import Viyog

    v = Viyog(model)            # attaches a hook to the first conv layer
    v.fit(id_loader)            # one ID pass: learn the dormant band + ID mean
    scores = v.score(loader)    # per-sample roughness; HIGHER => more adversarial
    v.close()

Or as a context manager (deterministic hook cleanup)::

    with Viyog(model) as v:
        v.fit(id_loader)
        adv_scores = v.score(adv_batch)

Direction convention (note it flips from magnitude-based detectors):

* **higher** ``V(x)`` -> more likely **ADV**
* **lower**  ``V(x)`` -> more likely **OOD** or ID

This module also ships :func:`viyog_metrics`, a small OOD/ADV separability report
(AUROC / AUPR / FPR@95 / AUTC). It needs ``scikit-learn`` — install the optional
extra with ``pip install viyog[metrics]``.
"""

from __future__ import annotations

import threading
import weakref
from collections.abc import Sequence
from typing import Any

import torch

_EPS = 1e-6  # matches the reference feature extractor (06b_extract_full.py)


class Viyog:
    """Dormant-band roughness detector for separating ADV from OOD inputs.

    The wrapper finds the model's first convolutional layer (preferring an
    attribute named ``conv1``) and registers a forward hook that captures the
    layer's output. :meth:`fit` runs one pass over in-distribution data to (a)
    rank channels by mean activation and select the quietest ``dorm_pct`` of the
    *alive* channels as the **dormant band**, and (b) record the ID mean of the
    roughness score. :meth:`score` then returns, per sample, the mean
    magnitude-normalised total variation over that dormant band — the Viyog
    statistic ``V(x)``.

    Args:
        model: The model to wrap. Its forward pass must traverse at least one
            ``Conv1d``/``Conv2d``/``Conv3d`` module (or expose an attribute named
            ``conv1``).
        device: Device for computation. If ``None`` (default) it is inferred from
            the model parameters on first use.
        layer: Optional explicit module (or dotted ``named_modules`` name) to hook
            instead of the auto-detected first conv layer.
        dorm_pct: Fraction of *alive* channels, taken from the quiet end, that form
            the dormant band. Default ``0.10`` (the paper's setting).
        dead_thresh: Channels whose ID mean absolute activation is at or below this
            value are treated as permanently dead and excluded from the dormant
            band (selecting them would make ``V(x)`` a constant 0 and the detector
            useless — this guards architectures such as DenseNet whose first conv
            has many dead channels).

    Attributes:
        dorm_idx_ (torch.Tensor | None): Indices of the dormant-band channels
            (set by :meth:`fit`).
        id_profile_ (torch.Tensor | None): Per-channel mean absolute activation on
            ID data (set by :meth:`fit`).
        id_score_mean_ (float | None): Mean of ``V(x)`` over the ID fit set; use it
            to centre scores or pick a threshold. ``None`` until fit completes.
        n_channels_ (int | None): Number of channels in the hooked layer's output.
        layer_name_ (str | None): Name of the hooked layer.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device | str | None = None,
        layer: torch.nn.Module | str | None = None,
        dorm_pct: float = 0.10,
        dead_thresh: float = 1e-4,
    ) -> None:
        if not 0.0 < dorm_pct <= 1.0:
            raise ValueError(f"dorm_pct must be in (0, 1], got {dorm_pct}")
        self.model = model
        self.device = torch.device(device) if device is not None else None
        self.dorm_pct = float(dorm_pct)
        self.dead_thresh = float(dead_thresh)

        # fitted state
        self.dorm_idx_: torch.Tensor | None = None
        self.id_profile_: torch.Tensor | None = None
        self.id_score_mean_: float | None = None
        self.n_channels_: int | None = None

        self._hook_handle: torch.utils.hooks.RemovableHandle | None = None
        self._features: dict[str, torch.Tensor] = {}
        # Guards the capture-forward-read sequence in _forward_features: the hook
        # writes into shared instance state, so two threads calling score() on the
        # same detector could otherwise read each other's activations.
        self._lock = threading.Lock()

        name, mod = self._resolve_layer(model, layer)
        if mod is None:
            raise RuntimeError("No convolutional layer found to attach hook to.")
        self.layer_name_ = name
        self._is_conv1d = isinstance(mod, torch.nn.Conv1d)

        # The hook closes over a *weak* reference, not self. A strong reference
        # would form a cycle (model -> _forward_hooks -> closure -> self), and
        # because the caller owns the model, a Viyog dropped without close()
        # would survive until the next cyclic-GC pass, leaving a live hook doing
        # real work on every forward.
        self_ref = weakref.ref(self)

        def hook_fn(module: Any, inputs: Any, output: Any) -> None:
            inst = self_ref()
            if inst is not None:
                inst._features["first"] = output.detach()

        self._hook_handle = mod.register_forward_hook(hook_fn)

    # ------------------------------------------------------------------ hooks
    @staticmethod
    def _resolve_layer(
        module: torch.nn.Module, layer: torch.nn.Module | str | None
    ) -> tuple[str | None, torch.nn.Module | None]:
        """Resolve the module to hook: explicit override, then ``conv1``, then the
        first ``Conv{1,2,3}d`` in ``named_modules`` order.
        """
        if isinstance(layer, torch.nn.Module):
            return layer.__class__.__name__, layer
        if isinstance(layer, str):
            for name, sub in module.named_modules():
                if name == layer:
                    return name, sub
            raise ValueError(f"No submodule named {layer!r} found.")
        if hasattr(module, "conv1") and isinstance(module.conv1, torch.nn.Module):
            return "conv1", module.conv1
        for name, sub in module.named_modules():
            if name == "":
                continue
            if isinstance(sub, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
                return name, sub
        return None, None

    def __enter__(self) -> Viyog:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        """Remove the forward hook. Call when the detector is no longer needed."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ---------------------------------------------------------------- helpers
    def _ensure_device(self) -> torch.device:
        if self.device is None:
            self.device = next((p.device for p in self.model.parameters()), torch.device("cpu"))
        return self.device

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run one forward pass and return the hooked first-layer activation.

        The clear-forward-read sequence is serialised: the hook deposits the
        activation in shared instance state, so concurrent calls on one detector
        would otherwise be able to read each other's batch.
        """
        with self._lock:
            self._features.pop("first", None)
            with torch.no_grad():
                self.model(x)
            feats = self._features.pop("first", None)
        if feats is None:
            raise RuntimeError("Hook captured no features; check the model forward path.")
        return feats

    @staticmethod
    def _mean_and_tv(feats: torch.Tensor, is_conv1d: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-channel mean |activation| and magnitude-normalised total variation.

        Accepts ``(B, C, H, W)`` conv maps (2-D total variation over H and W) or
        ``(B, C, L)`` maps (1-D total variation over L, for 1-D signal models).

        Returns:
            ``(fmean, tv)`` each shaped ``(B, C)``. ``tv`` is the average absolute
            difference between neighbouring positions, divided by ``fmean`` so it
            measures *shape* (roughness), not magnitude.
        """
        if feats.dim() == 3 and not is_conv1d:
            # Unbatched Conv2d output (C, H, W) -- PyTorch allows unbatched conv
            # input/output; a Conv2d hook can never legitimately produce a real
            # (B, C, L) 1-D-conv batch, so this is unambiguous.
            feats = feats.unsqueeze(0)
        if feats.dim() == 4:
            absa = feats.abs()
            fmean = absa.mean(dim=(2, 3))
            H, W = feats.shape[2], feats.shape[3]
            if H > 1 and W > 1:
                dh = (feats[:, :, 1:, :] - feats[:, :, :-1, :]).abs().mean(dim=(2, 3))
                dw = (feats[:, :, :, 1:] - feats[:, :, :, :-1]).abs().mean(dim=(2, 3))
                tv = (dh + dw) / (fmean + _EPS)
            else:
                tv = torch.zeros_like(fmean)
        elif feats.dim() == 3:
            absa = feats.abs()
            fmean = absa.mean(dim=2)
            if feats.shape[2] > 1:
                dl = (feats[:, :, 1:] - feats[:, :, :-1]).abs().mean(dim=2)
                tv = dl / (fmean + _EPS)
            else:
                tv = torch.zeros_like(fmean)
        else:
            raise ValueError(
                f"Expected a 3-D or 4-D activation map, got shape {tuple(feats.shape)}"
            )
        return fmean, tv

    @staticmethod
    def _batch_x(batch: Any) -> torch.Tensor:
        """Extract the input tensor from a ``(inputs, ...)`` batch or a raw tensor."""
        if isinstance(batch, (list, tuple)) and len(batch) >= 1:
            return batch[0]
        return batch

    # -------------------------------------------------------------------- fit
    @torch.no_grad()
    def fit(self, id_loader: torch.utils.data.DataLoader) -> Viyog:
        """Learn the dormant band and the ID roughness mean from ID data.

        Runs a single pass over ``id_loader`` accumulating the per-channel mean
        absolute activation (to rank channels and pick the dormant band) and the
        per-channel mean total variation (to record the ID mean of ``V(x)``).

        Args:
            id_loader: Iterable of in-distribution batches. Each batch may be a
                tensor or a ``(inputs, labels)`` tuple — only the inputs are used.

        Returns:
            ``self`` (so calls can be chained).

        Raises:
            RuntimeError: If the loader yields no batches.
        """
        device = self._ensure_device()
        self.model = self.model.to(device)
        self.model.eval()

        sum_fmean: torch.Tensor | None = None
        sum_tv: torch.Tensor | None = None
        count = 0

        for batch in id_loader:
            x = self._batch_x(batch).to(device)
            feats = self._forward_features(x)
            fmean, tv = self._mean_and_tv(feats, self._is_conv1d)
            if sum_fmean is None:
                sum_fmean = fmean.sum(dim=0).double()
                sum_tv = tv.sum(dim=0).double()
            else:
                sum_fmean += fmean.sum(dim=0).double()
                sum_tv += tv.sum(dim=0).double()
            count += fmean.shape[0]

        if count == 0 or sum_fmean is None:
            raise RuntimeError("id_loader produced no batches.")

        profile = (sum_fmean / count).float()  # (C,) mean |act| per channel
        mean_tv = (sum_tv / count).float()  # (C,) mean TV per channel
        C = profile.numel()
        self.n_channels_ = int(C)
        self.id_profile_ = profile

        alive = torch.nonzero(profile > self.dead_thresh, as_tuple=False).flatten()
        if alive.numel() == 0:
            alive = torch.arange(C, device=profile.device)
        alive_sorted = alive[torch.argsort(profile[alive])]  # quiet -> loud
        n_low = max(1, int(alive.numel() * self.dorm_pct))
        self.dorm_idx_ = alive_sorted[:n_low].clone()

        self.id_score_mean_ = float(mean_tv[self.dorm_idx_].mean().item())
        return self

    # ------------------------------------------------------------------ score
    @torch.no_grad()
    def score(
        self,
        x: torch.Tensor | torch.utils.data.DataLoader,
        center: bool = False,
    ) -> torch.Tensor:
        """Per-sample dormant-band roughness ``V(x)``.

        If ``x`` is a batch tensor, returns a 1-D tensor of scores for that batch.
        If ``x`` is a ``DataLoader`` (or any iterable of batches), scores the whole
        thing and returns the concatenation.

        Args:
            x: Input batch or data loader.
            center: If ``True``, subtract the ID mean recorded by :meth:`fit`
                (``id_score_mean_``). Centering is monotonic, so it does not change
                AUROC — it only shifts scores so that ID sits near 0.

        Returns:
            1-D ``torch.Tensor`` of scores on the inferred device. **Higher means
            more adversarial.**

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if self.dorm_idx_ is None:
            raise RuntimeError("Call fit() before score().")

        device = self._ensure_device()
        self.model = self.model.to(device)
        self.model.eval()

        if isinstance(x, torch.Tensor):
            feats = self._forward_features(x.to(device))
            _, tv = self._mean_and_tv(feats, self._is_conv1d)
            dorm = self.dorm_idx_.to(tv.device)
            v = tv.index_select(1, dorm).mean(dim=1)
            if center and self.id_score_mean_ is not None:
                v = v - self.id_score_mean_
            return v

        # iterable of batches / DataLoader
        out: list[torch.Tensor] = []
        for batch in x:
            out.append(self.score(self._batch_x(batch), center=center))
        return torch.cat(out) if out else torch.empty(0, device=device)

    def score_loader(
        self, loader: torch.utils.data.DataLoader, center: bool = False
    ) -> torch.Tensor:
        """Score an entire loader and return a single 1-D tensor. See :meth:`score`."""
        return self.score(loader, center=center)

    # --------------------------------------------------------- optional squash
    @staticmethod
    def bounded_score(scores: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Optional monotone squash of raw scores into ``(0, 1)``.

        A convenience for thresholding / display only. Because the transform is
        strictly monotonic, the AUROC of the squashed scores equals that of the raw
        scores. Not applied by :meth:`score`; call it explicitly if you want a
        bounded output.

        Args:
            scores: Raw (ideally ID-centered) ``V(x)`` scores.
            temperature: Larger values make the transition gentler.

        Returns:
            Tensor of the same shape with values in ``(0, 1)``.
        """
        return torch.sigmoid(scores / float(temperature))


# ---------------------------------------------------------------------------
# Metrics (optional: needs scikit-learn — `pip install viyog[metrics]`)
# ---------------------------------------------------------------------------
def viyog_metrics(
    ood_scores: Sequence[float],
    adv_scores: Sequence[float],
    recall_level: float = 0.95,
) -> dict[str, Any]:
    """Separability report for two score populations (default task: OOD vs ADV).

    Labels the second population as positive (``1``) and the first as negative
    (``0``), then reports threshold-free and threshold-based separability. With the
    Viyog convention (higher ``V(x)`` => more adversarial), pass ``ood_scores`` and
    ``adv_scores`` so that "positive" means ADV; the report is symmetric, so any two
    populations work (e.g. ID vs ADV).

    Args:
        ood_scores: Scores for the negative class (e.g. OOD or ID).
        adv_scores: Scores for the positive class (e.g. ADV).
        recall_level: TPR level at which to report FPR (default ``0.95``).

    Returns:
        Dict with ``AUROC``, ``AUPR_IN``, ``AUPR_OUT``, ``FPR95``,
        ``DetectionError``, ``AUTC`` and ``AUTC_components`` (``AUFPR``/``AUFNR``).

    Raises:
        ImportError: If scikit-learn is not installed.
    """
    try:
        import numpy as np
        from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
    except ImportError as exc:  # pragma: no cover - trivial guard
        raise ImportError(
            "viyog_metrics needs scikit-learn. Install it with: pip install 'viyog[metrics]'"
        ) from exc

    neg = np.asarray(ood_scores, dtype=np.float64)
    pos = np.asarray(adv_scores, dtype=np.float64)
    scores = np.concatenate([neg, pos])
    labels = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])

    auroc = roc_auc_score(labels, scores)
    aupr_out = average_precision_score(labels, scores)
    aupr_in = average_precision_score(1 - labels, -scores)

    fpr, tpr, thresholds = roc_curve(labels, scores)
    idx = int(np.searchsorted(tpr, recall_level))
    fpr95 = float(fpr[min(idx, len(fpr) - 1)])
    det_error = float(np.min(0.5 * (fpr + (1 - tpr))))

    # AUTC (pytorch-ood style). sklearn prepends an infinite threshold; drop the
    # non-finite entries so the trapezoidal integral stays finite.
    finite = np.isfinite(thresholds)
    th, f_, t_ = thresholds[finite], fpr[finite], tpr[finite]
    if len(th) > 1 and th[0] > th[-1]:
        th, f_, t_ = th[::-1], f_[::-1], t_[::-1]
    aufpr = float(np.trapezoid(f_, th)) if len(th) > 1 else 0.0
    aufnr = float(np.trapezoid(1.0 - t_, th)) if len(th) > 1 else 0.0
    autc = 0.5 * (aufpr + aufnr)

    return {
        "AUROC": float(auroc),
        "AUPR_IN": float(aupr_in),
        "AUPR_OUT": float(aupr_out),
        "FPR95": fpr95,
        "DetectionError": det_error,
        "AUTC": float(autc),
        "AUTC_components": {"AUFPR": aufpr, "AUFNR": aufnr},
    }
