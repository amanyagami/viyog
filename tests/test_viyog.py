import torch
from torch.utils.data import DataLoader, TensorDataset

from viyog import Viyog


class _SmallConvNet(torch.nn.Module):
    """Tiny model with attribute `conv1` so Viyog finds it predictably."""

    def __init__(self) -> None:
        super().__init__()
        # conv1 will be discovered by Viyog
        self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)


def make_random_loader(total: int = 8, batch_size: int = 4) -> DataLoader:
    # simple inputs shaped (N, 3, 8, 8)
    inputs = torch.randn(total, 3, 8, 8)
    labels = torch.zeros(total, dtype=torch.long)
    ds = TensorDataset(inputs, labels)
    return DataLoader(ds, batch_size=batch_size)


def test_hook_captures_features() -> None:
    model = _SmallConvNet()
    v = Viyog(model)
    # single batch
    x = torch.randn(2, 3, 8, 8)
    feats = v._get_first_layer_features(x)
    assert isinstance(feats, torch.Tensor)
    # conv1 output channels = 4
    assert feats.ndim >= 4
    assert feats.shape[1] == 4
    # detached (no grad)
    assert not feats.requires_grad
    v.close()


def test_fit_and_score_tensor() -> None:
    model = _SmallConvNet()
    v = Viyog(model)
    loader = make_random_loader(total=8, batch_size=4)
    mean = v.fit(loader)
    assert isinstance(mean, float)
    assert mean >= 0.0

    # score a single batch tensor
    batch = torch.randn(3, 3, 8, 8)
    scores = v.score(batch, Temperature=100.0)
    assert isinstance(scores, torch.Tensor)
    assert scores.shape[0] == 3
    # scores in reasonable numeric range (-1, 1)
    assert torch.all(scores <= 1.0 + 1e-6)
    assert torch.all(scores >= -1.0 - 1e-6)
    v.close()


def test_score_loader_concat() -> None:
    model = _SmallConvNet()
    v = Viyog(model)
    loader = make_random_loader(total=10, batch_size=3)  # yields 4 batches (3+3+3+1)
    v.fit(make_random_loader(total=6, batch_size=3))
    out = v.score(loader)
    # total samples should match
    total = sum(
        len(batch[0]) if isinstance(batch, (list, tuple)) else len(batch) for batch in loader
    )
    assert out.shape[0] == total
    v.close()


def test_context_manager_and_hook_removal() -> None:
    model = _SmallConvNet()
    # use context manager to ensure hook is removed on exit
    with Viyog(model) as v:
        x = torch.randn(2, 3, 8, 8)
        _ = v._get_first_layer_features(x)
        # features should be present immediately after forward
        assert "first" in v._features

    # after exiting context, the hook should be removed
    # run a forward; _features should not be populated
    v._features.pop("first", None)
    _ = model(torch.randn(2, 3, 8, 8))
    assert "first" not in v._features
