from dataclasses import dataclass
from typing import NamedTuple, TypeAlias

import torch
import torch.nn.functional as F

Tensor: TypeAlias = torch.Tensor


def _compute_autocorrelation_scipy(X: Tensor):
    import scipy

    X = X.squeeze()
    assert X.dim() == 1

    # TODO: fix for odd N
    assert len(X) % 2 == 0
    n = len(X) // 2

    X = X - X.mean()

    # Γ(t) is the correlation function
    Γ = torch.from_numpy(scipy.signal.correlate(X, X, mode="same"))

    # Normalise s.t. Γ(0) = 1 and take +ve `t` only
    Γ = Γ[n:] / Γ[n]

    return Γ


def compute_autocorrelation_scipy(X: Tensor):
    assert X.squeeze().dim() <= 2
    return torch.stack(
        [_compute_autocorrelation_scipy(replica) for replica in X]
    )


def _compute_autocorrelation_torch(X: Tensor):
    X = X.squeeze()
    assert X.dim() == 1
    n = len(X) // 2

    X = X - X.mean()

    Γ = F.conv1d(X.view(1, 1, -1), X.view(1, 1, -1), padding="same").squeeze()

    Γ = Γ[n - 1 :] / Γ[n - 1]  # noqa E203

    return Γ


compute_autocorrelation_torch = torch.vmap(_compute_autocorrelation_torch)


class IntegratedAutocorrelationErrors(NamedTuple):
    stat: Tensor
    bias: Tensor
    grad_stat: Tensor
    grad_bias: Tensor


@dataclass(frozen=True)
class ComputedAutocorrelations:
    autocorrelation: Tensor
    integrated: float
    truncation_window: int
    errors: IntegratedAutocorrelationErrors


def compute_autocorrelations(
    X: Tensor, λ: float = 2.0
) -> ComputedAutocorrelations:
    X = torch.atleast_2d(X.squeeze())
    assert X.dim() == 2

    N = X.shape[1]  # sample size
    W = torch.arange(N // 2)  # compute window

    try:
        Γ = compute_autocorrelation_scipy(X)
    except ImportError:
        Γ = compute_autocorrelation_torch(X)

    # Average over replica
    # WARNING uses nanmean, which might hide issues
    Γ = Γ.nanmean(dim=0)

    # The summation window-dependent integrated autocorrelation
    τ_int = Γ.cumsum(dim=0) - 0.5

    # The associated exponential autocorrelation time
    τ_exp = ((2 * τ_int - 1) / (2 * τ_int + 1)).log().reciprocal().negative()
    τ_exp = τ_exp.nan_to_num().clamp(min=1e-6)

    # Statistical error (Eq. 42 in arxiv.org/pdf/hep-lat/0306017)
    ε_stat = torch.sqrt((4 / N) * (W + 1 / 2 - τ_int)) * τ_int

    # Truncation bias
    ε_bias = -τ_int * torch.exp(-W / (λ * τ_exp))

    # λ times W-derivative of the errors
    dεdW_stat = τ_exp / torch.sqrt(N * W)
    dεdW_bias = (-1 / λ) * torch.exp(-W / (λ * τ_exp))

    errors = IntegratedAutocorrelationErrors(
        stat=ε_stat,
        bias=ε_bias,
        grad_stat=dεdW_stat,
        grad_bias=dεdW_bias,
    )

    # Derivative of the sum of absolute errors
    dεdW = errors.grad_stat + errors.grad_bias

    # argmax returns first occurrence of the derivative being positive,
    # indicating that the total error will increase for larger window sizes
    W_opt = torch.argmax((dεdW[1:] > 0).int(), dim=0) + 1

    # return τ_int.gather(dim=1, index=W_opt.unsqueeze(1))

    # Select the best estimate of the integrated autocorrelation time
    τ_int = τ_int[W_opt].item()

    return ComputedAutocorrelations(
        autocorrelation=Γ,
        integrated=τ_int,
        truncation_window=W_opt,
        errors=errors,
    )
