import random

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")

import numpy as np
import torch

from src.utils.reproducibility import set_seed


def test_set_seed_reproducible_values() -> None:
    set_seed(7)
    py_first = random.random()
    np_first = np.random.rand()
    torch_first = torch.rand(1).item()

    set_seed(7)
    assert py_first == random.random()
    assert np_first == np.random.rand()
    assert torch_first == torch.rand(1).item()
