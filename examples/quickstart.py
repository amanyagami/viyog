"""Viyog quickstart — ID vs OOD vs ADV separation, end to end.

Loads a real model, crafts PGD adversarials, and uses the packaged ``viyog.Viyog``
detector to score clean in-distribution (ID), out-of-distribution (OOD), and
adversarial (ADV) inputs. Prints the three separation AUROCs.

    pip install "viyog[metrics]"              # --smoke needs nothing else
    pip install "viyog[metrics]" torchvision  # only needed for the real-data path

    python examples/quickstart.py          # real: downloads CIFAR-10 + SVHN + ResNet18 weights
    python examples/quickstart.py --smoke  # synthetic data + a tiny CNN, no torchvision needed

Expected shape of the result: ADV scores far above ID/OOD (strong ID-vs-ADV
separation), a moderate OOD-vs-ADV gap, and weak ID-vs-OOD (Viyog is an ADV
specialist — pair it with a logit OOD score for the ID-vs-OOD axis).
"""

from __future__ import annotations

import argparse
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from viyog import Viyog, viyog_metrics

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Normalized(nn.Module):
    """Wrap a backbone so it accepts [0,1] images and normalises internally.

    Passing this whole module to Viyog means both the attack and the detector
    operate in raw [0,1] pixel space, while the backbone still sees normalised
    inputs. Viyog auto-detects the backbone's first conv (``.conv1``).
    """

    def __init__(self, model: nn.Module, mean: tuple, std: tuple) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self.mean) / self.std)


class _TinySmokeCNN(nn.Module):
    """Stand-in for ResNet18 in --smoke mode, so it needs no torchvision.

    conv1 matches ResNet18's real first-conv shape (Conv2d(3, 64, 7, stride=2,
    padding=3)), so Viyog hooks a representative first layer even though
    everything past it is a minimal random-init classifier head.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 1000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        return self.fc(self.pool(x).flatten(1))


def pgd(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float, alpha: float, steps: int
) -> torch.Tensor:
    """Standard L-infinity PGD (Madry 2018) in [0,1] pixel space."""
    x_adv = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = (x_adv + alpha * grad.sign()).detach()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0, 1)
    return x_adv


def get_images(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (id_images, ood_images), each (n, 3, size, size) in [0,1]."""
    if args.smoke:  # smooth low-frequency images, no download

        def smooth(n: int) -> torch.Tensor:
            coarse = torch.randn(n, 3, args.size // 8, args.size // 8)
            up = F.interpolate(coarse, size=args.size, mode="bilinear", align_corners=False)
            return (up - up.amin()) / (up.amax() - up.amin() + 1e-6)

        return smooth(args.n), smooth(args.n) * 0.6 + 0.2

    import torchvision as tv
    import torchvision.transforms as T

    tf = T.Compose([T.Resize(args.size), T.CenterCrop(args.size), T.ToTensor()])
    id_ds = tv.datasets.CIFAR10(args.data_root, train=False, download=True, transform=tf)
    ood_ds = tv.datasets.SVHN(args.data_root, split="test", download=True, transform=tf)
    id_x = torch.stack([id_ds[i][0] for i in range(args.n)])
    ood_x = torch.stack([ood_ds[i][0] for i in range(args.n)])
    return id_x, ood_x


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n", type=int, default=512, help="samples per split")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--eps", type=float, default=8 / 255, help="PGD L-inf budget")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--smoke", action="store_true", help="synthetic data, no downloads")
    args = ap.parse_args()
    dev = args.device

    if args.smoke:
        backbone = _TinySmokeCNN()
    else:
        import torchvision as tv

        backbone = tv.models.resnet18(weights=tv.models.ResNet18_Weights.DEFAULT)
    model = Normalized(backbone, IMAGENET_MEAN, IMAGENET_STD)
    model = model.to(dev).eval()

    id_all, ood_x = get_images(args)
    id_all, ood_x = id_all.to(dev), ood_x.to(dev)
    fit_x, id_x = id_all[: args.n // 2], id_all[args.n // 2 :]  # fit on one half, score the other

    # craft adversarials on the score-half (untargeted PGD vs the model's own labels)
    with torch.no_grad():
        y = model(id_x).argmax(1)
    adv_x = pgd(model, id_x, y, args.eps, args.eps / 4, args.steps)

    with Viyog(model, device=dev) as v:
        v.fit([(fit_x,)])  # learn the dormant band on clean ID
        s_id = v.score(id_x).cpu().numpy()
        s_ood = v.score(ood_x).cpu().numpy()
        s_adv = v.score(adv_x).cpu().numpy()
        print(
            f"hooked layer: {v.layer_name_} | channels: {v.n_channels_} | "
            f"dorm band: {v.dorm_idx_.numel()}"
        )

    def auroc(neg: Any, pos: Any) -> float:
        # directionless separability (the paper's convention): orientation is fixed
        # from the fit set, so fold the AUROC to be >= 0.5.
        a = viyog_metrics(neg, pos)["AUROC"]
        return max(a, 1.0 - a)

    if args.smoke:
        print(
            "[smoke] synthetic data — a code sanity-check only, not representative "
            "of real AUROCs (esp. T1). Run without --smoke for real numbers."
        )
    print(
        f"\nmean V(x):  ID={s_id.mean():.3f}  OOD={s_ood.mean():.3f}  "
        f"ADV={s_adv.mean():.3f}   (higher => more adversarial)"
    )
    print(f"T2  ID  vs ADV : AUROC={auroc(s_id, s_adv):.3f}   (adversarial detection)")
    print(f"T3  OOD vs ADV : AUROC={auroc(s_ood, s_adv):.3f}   (OOD-vs-ADV separation)")
    print(f"T1  ID  vs OOD : AUROC={auroc(s_id, s_ood):.3f}   (weak; use a logit score)")


if __name__ == "__main__":
    main()
