import math
from math import pi as π

import pytest
import torch

from torchlft.observables import IntegratedAutocorrelation


def test_autocorrelation():
    x = torch.linspace(0, 10, 100).negative().exp() + torch.randn(100) / 5
    auto = IntegratedAutocorrelation(x)
    tau = auto.compute()
