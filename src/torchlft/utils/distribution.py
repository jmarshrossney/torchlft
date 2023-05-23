"""
A collection of utils for constructing objects based on distributions.
"""
from types import MethodType

import torch
import torch.nn as nn
import torch.distributions
import pytorch_lightning as pl

from torchlft.utils.tensor import sum_except_batch

from torchlft.typing import Tensor


def expand_iid(
    distribution: torch.distributions.Distribution,
    extra_dims: torch.Size,
) -> torch.distributions.Independent:
    """
    Constructs a multivariate distribution with iid components.

    The resulting distribution is interpreted as the joint distribution
    comprising `n` copies of the input distribution, where `n` is the
    number of components implied by `extra_dims`.

    The `event_shape` of the joint distribution is
    `(*extra_dims, *existing_dims)`, where `existing_dims` are inherited
    from the original distribution.

    Args:
        distribution:
            The distribution of the iid components
        extra_dims:
            Sizes of additional dimensions to prepend

    Returns:
        A multivariate distibution with independent components

    Example:

        This will create a Multivariate Gaussian with diagonal covariance,
        with an `event_shape` of [6, 6]

        >>> # Create a univariate Gaussian distribution
        >>> uv_gauss = torch.distributions.Normal(0, 1)
        >>> uv_gauss.batch_shape, uv_gauss.event_shape
        (torch.Size([]), torch.Size([]))
        >>> uv_gauss.sample().shape
        torch.Size([])
        >>>
        >>> # Create a multivariate Gaussian with diagonal covariance
        >>> mv_gauss = expand_iid(uv_gauss, [6, 6])
        >>> mv_gauss.batch_shape, mv_gauss.event_shape
        (torch.Size([]), torch.Size([6, 6])
        >>> mv_gauss.sample().shape
        torch.Size([6, 6])
        >>>
        >>> # What happens when we compute the log-prob?
        >>> uv_gauss.log_prob(mv_gauss.sample()).shape
        torch.Size([6, 6])
        >>> mv_gauss.log_prob(mv_gauss.sample()).shape
        torch.Size([])

        This will create a Multivariate Gaussian comprising 6x6
        copies of an original Bivariate Gaussian, resulting in
        a 6x6x2 dimensional distribution

        >>> # Create a bivariate Gaussian distribution
        >>> bv_gauss = torch.distributions.MultivariateNormal(
                loc=torch.rand(2),
                scale_tril=torch.eye(2) + torch.rand(2, 2).tril(),
        >>> bv_gauss.batch_shape, uv_gauss.event_shape
        (torch.Size([]), torch.Size([2]))
        >>> bv_gauss.sample().shape
        torch.Size([2])
        >>>
        >>> # Create a 6x6x2 multivariate Gaussian
        >>> mv_gauss = expand_iid(bv_gauss, [6, 6])
        >>> mv_gauss.batch_shape, mv_gauss.event_shape
        (torch.Size([]), torch.Size([6, 6, 2])
        >>> mv_gauss.sample().shape
        torch.Size([6, 6, 2])
    """
    # Expand original distribution by lattice_shape
    distribution = distribution.expand(extra_dims)

    # Register the components as being part of one distribution
    distribution = torch.distributions.Independent(
        distribution, reinterpreted_batch_ndims=len(distribution.batch_shape)
    )

    return distribution


class DistributionModule(nn.Module):
    def __init__(
        self,
        cls: type[torch.distributions.Distribution],
        **kwargs,
    ):
        super().__init__()
        for key, val in kwargs.items():
            self.register_buffer(key, torch.as_tensor(val))

        params = {key: getattr(self, key) for key, val in kwargs.items()}
        self.distribution = cls(**params)


class DiagonalGaussianModule(DistributionModule):
    def __init__(self, shape: torch.Size):
        super().__init__(torch.distributions.Normal, loc=0.0, scale=1.0)
        self.distribution = expand_iid(
            self.distribution,
            extra_dims=shape,
        )


class GaussianModule(DistributionModule):
    def __init__(
        self,
        mean: Tensor,
        *,
        covariance: Tensor | None = None,
        precision: Tensor | None = None,
    ):
        assert (covariance is not None) ^ (precision is not None)
        kwargs = {
            "covariance_matrix": covariance,
            "precision_matrix": precision,
            "loc": mean,
        }
        super().__init__(torch.distributions.MultivariateNormal, **kwargs)
