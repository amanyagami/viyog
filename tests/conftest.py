# make the src/ layout importable (adjust if your layout differs)
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def deterministic_seed() -> None:
    """Deterministic RNG for tests that use random numbers."""
    seed = 12345
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
