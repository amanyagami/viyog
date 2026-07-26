"""Model loading utilities.

Handles:
- timm model creation with correct num_classes
- Loading custom .pth checkpoints (supports bare state-dict or wrapped ckpt)
- NormalizedModel: wraps any model so it normalises [0,1] inputs internally,
  allowing torchattacks to operate cleanly in [0,1] pixel space.
- First-conv-layer discovery (needed for Viyog + feature hooks).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import timm
import torch
import torch.nn as nn

from config import DEVICE, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES


class NormalizedModel(nn.Module):
    """Wraps a model to normalise [0,1] float inputs before the backbone.

    This keeps torchattacks working cleanly in pixel space: attacks receive
    and output valid [0,1] tensors while the backbone always sees properly
    normalised inputs.

    Parameters
    ----------
    model:  backbone (expects normalised inputs)
    mean:   per-channel mean, length 3
    std:    per-channel std, length 3
    """

    def __init__(
        self,
        model: nn.Module,
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
    ) -> None:
        super().__init__()
        self.model = model
        # Register as buffers → moves with .to(device)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self.mean) / self.std)


def _load_state_dict(path: Path, device: str) -> dict[str, torch.Tensor]:
    """Load a checkpoint file and extract the state dict.

    Handles the HuggingFace amanyagami/Cifar100_Finetuned format where
    checkpoints are saved as:
        {"model_state": <state_dict>, "optimizer_state": ...,
         "epoch": int, "val_acc": float, "model_name": str}
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        # Log reported val_acc if present (useful sanity check)
        if "val_acc" in ckpt:
            print(f"  Checkpoint val_acc={ckpt['val_acc']:.4f}  epoch={ckpt.get('epoch', '?')}")
        for key in ("model_state", "model", "state_dict", "model_state_dict", "net"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        # Bare state dict (all values are tensors)
        return ckpt
    return ckpt


def _strip_prefix(state_dict: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Remove a key prefix (e.g. 'module.' from DataParallel)."""
    return {
        (k[len(prefix):] if k.startswith(prefix) else k): v
        for k, v in state_dict.items()
    }


def load_model(
    arch: str,
    weight_path: Path | None,
    num_classes: int = NUM_CLASSES,
    device: str = DEVICE,
) -> nn.Module:
    """Create a timm model and optionally load custom weights.

    Parameters
    ----------
    arch:         timm model name (e.g. 'swin_tiny_patch4_window7_224')
    weight_path:  path to .pth checkpoint; None → timm pretrained weights
    num_classes:  output classes (100 for CIFAR-100)
    device:       torch device string

    Returns
    -------
    model set to eval() and moved to device
    """
    pretrained = weight_path is None
    model = timm.create_model(arch, pretrained=pretrained, num_classes=num_classes)

    if weight_path is not None:
        if not weight_path.exists():
            raise FileNotFoundError(f"Weight file not found: {weight_path}")
        sd = _load_state_dict(weight_path, device)
        # Strip exactly one wrapper prefix (DataParallel / Lightning / custom).
        # Break after the first match to avoid chaining strips that corrupt keys
        # (e.g. "model.backbone.xxx" → only strip "model.", leave "backbone.xxx").
        for prefix in ("module.", "model.", "backbone.", "encoder."):
            if any(k.startswith(prefix) for k in sd):
                sd = _strip_prefix(sd, prefix)
                break
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"  [warn] {arch}: {len(missing)} missing keys: {missing[:3]}")
        if unexpected:
            print(f"  [warn] {arch}: {len(unexpected)} unexpected keys: {unexpected[:3]}")

    model = model.to(device).eval()
    return model


def load_normalized_model(
    arch: str,
    weight_path: Path | None,
    num_classes: int = NUM_CLASSES,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    device: str = DEVICE,
) -> NormalizedModel:
    """Load model and wrap it with NormalizedModel."""
    backbone = load_model(arch, weight_path, num_classes=num_classes, device=device)
    wrapper = NormalizedModel(backbone, mean=mean, std=std).to(device).eval()
    return wrapper


def find_first_conv(module: nn.Module) -> tuple[str | None, nn.Module | None]:
    """Return (name, layer) of the first Conv1d/2d/3d in the module.

    Prefers an attribute named 'conv1' if present; otherwise iterates
    named_modules() in order.
    """
    if hasattr(module, "conv1") and isinstance(
        module.conv1, (nn.Conv1d, nn.Conv2d, nn.Conv3d)
    ):
        return "conv1", module.conv1

    for name, sub in module.named_modules():
        if name == "":
            continue
        if isinstance(sub, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            return name, sub
    return None, None


def find_first_conv_in_normalized(wrapper: NormalizedModel) -> tuple[str | None, nn.Module | None]:
    """Find first conv in the backbone of a NormalizedModel."""
    return find_first_conv(wrapper.model)


class FirstLayerHook:
    """Context-manager that captures the first-conv output during forward passes.

    Usage::

        with FirstLayerHook(norm_model) as hook:
            logits = norm_model(x)
            feats = hook.features   # (B, C, H, W) detached on same device
    """

    def __init__(self, norm_model: NormalizedModel) -> None:
        self._model = norm_model
        self.features: torch.Tensor | None = None
        self._handle: torch.utils.hooks.RemovableHandle | None = None

        _, layer = find_first_conv_in_normalized(norm_model)
        if layer is None:
            raise RuntimeError("No convolutional layer found in model.")
        self._layer = layer

    def __enter__(self) -> "FirstLayerHook":
        def _hook(module: nn.Module, inp: Any, output: torch.Tensor) -> None:
            self.features = output.detach()

        self._handle = self._layer.register_forward_hook(_hook)
        return self

    def __exit__(self, *args: Any) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def close(self) -> None:
        self.__exit__()
