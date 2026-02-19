"""Viyog wrapper + metrics.

This module provides the :class:`Viyog` wrapper that attaches a forward hook to the
first convolutional layer of a model to capture early-layer activations, an API to
fit a per-sample activation norm baseline, and scoring utilities for OOD detection.

Recommended usage:
    v = Viyog(model)
    v.fit(trainloader)
    scores = v.score(batch_or_loader)
    v.close()

Or use as a context manager:

    with Viyog(model) as v:
        v.fit(trainloader)
        scores = v.score(batch)

"""

from __future__ import annotations

import math
from typing import Tuple, Dict, Optional, List, Union

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


class Viyog:
    """
    Wrapper that captures first-layer activations and computes an activation-norm
    based score.

    The wrapper finds the first convolutional layer of `model` (prefers an attribute
    named ``conv1`` if present) and registers a forward hook that saves the layer's
    output for later inspection. Use :meth:`fit` to compute a training baseline
    (mean infinity-norm of per-sample flattened activations). Then call
    :meth:`score` or :meth:`score_loader` to compute Viyog scores on new data.

    Parameters
    ----------
    model : torch.nn.Module
        The model to wrap. The forward pass must traverse at least one Conv{1,2,3}d
        module (or provide an attribute named ``conv1``).
    device : torch.device | str | None, optional
        Device to use for computation. If ``None`` (default) the device is inferred
        from model parameters as needed.

    Attributes
    ----------
    id_norm_scores_mean : float or None
        Mean per-sample infinity norm computed by :meth:`fit`. ``None`` until fit
        completes successfully.
    """

    def __init__(self, model: torch.nn.Module, device: Optional[torch.device | str] = None):
        self.model = model
        # device may be provided or inferred later
        self.device = torch.device(device) if device is not None else None

        # state
        self.id_norm_scores_mean: Optional[float] = None
        self._hook_layer_name: Optional[str] = None
        self._hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        # will hold last-forward features (detached) while a forward is happening
        self._features: Dict[str, torch.Tensor] = {}

        # find conv & attach hook immediately
        name, layer = self._find_first_conv(self.model)
        if layer is None:
            raise RuntimeError("No convolutional layer found to attach hook to.")
        self._hook_layer_name = name

        # attach hook that stores a detached tensor on the device of the layer
        def hook_fn(module, input, output):
            # detach to avoid retaining graph; keep on same device for speed
            self._features["first"] = output.detach()

        self._hook_handle = layer.register_forward_hook(hook_fn)

    # Context-manager helpers so user can rely on deterministic cleanup
    def __enter__(self) -> "Viyog":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        """Remove the hook (call when you no longer need the Viyog wrapper)."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    @staticmethod
    def Viyog_Score(num: torch.Tensor, Temperature: float = 1000.0) -> torch.Tensor:
        """
        Convert centered norms into bounded scores in approximately (-1, 1).

        Parameters
        ----------
        num : torch.Tensor
            Tensor (any shape) of **centered** norms (i.e., per-sample norm minus
            training mean). Must be on the same device you want the result on.
        Temperature : float, optional
            Temperature scaling factor. Default is 1000.0.

        Returns
        -------
        torch.Tensor
            Tensor of same shape as ``num`` with values approx in (-1, 1).
        """
        num = num / float(Temperature)
        sign = torch.sign(num)
        num = torch.exp(torch.abs(num))
        denom = 1.0 + torch.exp(-num)
        return sign / denom

    @staticmethod
    def _find_first_conv(module: torch.nn.Module) -> Tuple[Optional[str], Optional[torch.nn.Module]]:
        """
        Find the first convolutional submodule.

        Prefers attribute ``conv1`` if present; otherwise iterates submodules in
        ``named_modules()`` order and returns the first instance of Conv1d/2d/3d.

        Returns
        -------
        (name, module) or (None, None) if not found
        """
        if hasattr(module, "conv1"):
            return "conv1", getattr(module, "conv1")
        for name, sub in module.named_modules():
            if name == "":
                continue
            if isinstance(sub, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
                return name, sub
        return None, None

    def _ensure_device(self) -> torch.device:
        """Infer and set :pyattr:`device` if not provided; return the device."""
        if self.device is None:
            for p in self.model.parameters():
                self.device = p.device
                break
            if self.device is None:
                # fallback to CPU if model has no params
                self.device = torch.device("cpu")
        return self.device

    def _get_first_layer_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run a forward and return the detached features captured by the hook.

        Notes
        -----
        - This uses the persistent hook attached in :meth:`__init__`, which writes
          to ``self._features`` under the key ``"first"``.
        - The model forward is executed under ``torch.no_grad()`` here to avoid
          allocating gradients.
        """
        # clear any previous feature
        self._features.pop("first", None)
        # forward (we assume user won't need gradients)
        with torch.no_grad():
            _ = self.model(x)
        if "first" not in self._features:
            raise RuntimeError("Hook did not capture features. Check model forward path.")
        return self._features["first"]

    @torch.no_grad()
    def fit(self, trainloader: torch.utils.data.DataLoader) -> float:
        """
        Compute the mean of per-sample infinity norms of the first-layer activations
        across the provided trainloader and store it as a Python float.

        Parameters
        ----------
        trainloader : torch.utils.data.DataLoader
            DataLoader providing samples to estimate the mean activation norm. Each
            yielded batch should be (inputs, labels) or where inputs are the first item.

        Returns
        -------
        float
            The computed mean per-sample infinity norm (stored in
            ``self.id_norm_scores_mean``).
        """
        device = self._ensure_device()
        self.model = self.model.to(device)
        self.model.eval()

        total_sum = 0.0
        total_count = 0
        any_batch = False

        for batch in trainloader:
            any_batch = True
            # support (inputs, labels) or just inputs
            if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                x = batch[0]
            else:
                x = batch
            x = x.to(device)
            feats = self._get_first_layer_features(x)  # detached tensor
            B = feats.shape[0]
            flat = feats.reshape(B, -1)
            batch_norms = torch.linalg.norm(flat, ord=float("inf"), dim=1)  # (B,)
            # convert to float64 on CPU for stable accumulation
            batch_norms = batch_norms.cpu().double()
            total_sum += float(batch_norms.sum().item())
            total_count += B

        if not any_batch:
            raise RuntimeError("Trainloader produced no batches.")
        mean = float(total_sum / total_count)
        self.id_norm_scores_mean = mean
        return self.id_norm_scores_mean

    @torch.no_grad()
    def score(self, x: Union[torch.Tensor, torch.utils.data.DataLoader], Temperature: float = 1000.0) -> torch.Tensor:
        """
        If `x` is a Tensor (batch): returns a 1D tensor of scores for that batch.
        If `x` is a DataLoader: processes entire loader and returns concatenated scores.

        Parameters
        ----------
        x : torch.Tensor or torch.utils.data.DataLoader
            Input batch (tensor) or data loader to score.
        Temperature : float, optional
            Temperature scaling factor passed through to :meth:`Viyog_Score`.

        Returns
        -------
        torch.Tensor
            1D tensor of scores (device equals inferred device).
        """
        if self.id_norm_scores_mean is None:
            raise RuntimeError("Call fit() before score(). id_norm_scores_mean is not set.")

        device = self._ensure_device()
        self.model = self.model.to(device)
        self.model.eval()

        if isinstance(x, torch.utils.data.DataLoader):
            # convenience: score an entire loader
            out = []
            for batch in x:
                if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                    xb = batch[0]
                else:
                    xb = batch
                out.append(self.score(xb, Temperature=Temperature))
            return torch.cat(out) if len(out) else torch.empty(0, device=device)

        # x is a Tensor (batch)
        xb = x.to(device)
        feats = self._get_first_layer_features(xb)
        B = feats.shape[0]
        flat = feats.reshape(B, -1)
        batch_norms = torch.linalg.norm(flat, ord=float("inf"), dim=1)  # (B,) on device
        # center by training mean (float)
        centered = batch_norms - float(self.id_norm_scores_mean)
        scores = self.Viyog_Score(centered, Temperature=Temperature)
        return scores

    def score_loader(self, loader: torch.utils.data.DataLoader, Temperature: float = 1000.0) -> torch.Tensor:
        """Helper that returns a single 1D tensor of scores for the whole loader."""
        return self.score(loader, Temperature=Temperature)

    def __del__(self):
        # ensure hook removed on deletion (best-effort)
        try:
            self.close()
        except Exception:
            pass


# ---------------- Viyog Metrics ----------------
def viyog_metrics(id_scores, ood_scores, recall_level: float = 0.95) -> dict:
    """
    Compute a collection of OOD detection metrics from id and ood scores.

    Parameters
    ----------
    id_scores : array-like
        Scores for in-distribution examples (lower means ID in the code's labeling).
    ood_scores : array-like
        Scores for OOD examples.
    recall_level : float, optional
        TPR level for which to compute FPR (default 0.95).

    Returns
    -------
    dict
        Dictionary with keys "AUROC", "AUPR_IN", "AUPR_OUT", "FPR95", "DetectionError",
        "AUTC" and "AUTC_components".
    """
    id_scores = np.asarray(id_scores)
    ood_scores = np.asarray(ood_scores)

    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([
        np.zeros(len(id_scores)),  # in-distribution -> label 0
        np.ones(len(ood_scores))   # out-of-distribution -> label 1
    ])

    auroc = roc_auc_score(labels, scores)
    aupr_out = average_precision_score(labels, scores)
    aupr_in = average_precision_score(1 - labels, -scores)

    fpr, tpr, thresholds = roc_curve(labels, scores)
    # FPR@95%TPR
    fpr95 = fpr[np.searchsorted(tpr, recall_level)]
    det_error = np.min(0.5 * (fpr + (1 - tpr)))

    # ---------------- AUTC (pytorch-ood style) ----------------
    # sklearn returns thresholds in decreasing order -> flip to ascending
    if thresholds[0] > thresholds[-1]:
        thresholds = thresholds[::-1]
        fpr = fpr[::-1]
        tpr = tpr[::-1]

    # area under FPR vs threshold
    aufpr = float(np.trapz(fpr, thresholds))
    # area under (1 - TPR) vs threshold
    aufnr = float(np.trapz(1.0 - tpr, thresholds))
    autc = 0.5 * (aufpr + aufnr)

    return {
        "AUROC": float(auroc),
        "AUPR_IN": float(aupr_in),
        "AUPR_OUT": float(aupr_out),
        "FPR95": float(fpr95),
        "DetectionError": float(det_error),
        "AUTC": float(autc),
        "AUTC_components": {
            "AUFPR": float(aufpr),
            "AUFNR": float(aufnr)
        }
    }
