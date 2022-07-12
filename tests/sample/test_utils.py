import math
import torch

from torchlft.sample.utils import autocorrelation


def test_autocorrelation():
    a = autocorrelation(torch.randn(10))
    assert a.shape == torch.Size([10])
    assert math.isclose(a[0], 1)
