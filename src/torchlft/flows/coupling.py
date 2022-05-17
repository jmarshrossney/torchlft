from __future__ import annotations

from typing import Optional

import torch

import torchlft.transforms


class CouplingLayer(torch.nn.Module):
    def __init__(
        self,
        transform: torchlft.transforms.Transform,
        transform_mask: torch.BoolTensor,
        condition_mask: Optional[torch.BoolTensor] = None,
    ) -> None:
        super().__init__()

        if condition_mask is None:
            # Then transformation conditioned on all non-transformed elements
            condition_mask = ~transform_mask

        assert torch.any(
            transform_mask
        ), "At least one element must be transformed!"
        assert torch.any(
            condition_mask
        ), "Transformation must be conditioned on at least one element!"
        assert not torch.any(
            torch.logical_and(transform_mask, condition_mask)
        ), "Transform_mask and condition_mask should not intersect!"
        self.register_buffer("_transform_mask", transform_mask)
        self.register_buffer("_condition_mask", condition_mask)

        self._transform = transform
        identity_params = torch.stack(
            [
                param.expand_as(transform_mask)
                for param in transform.identity_params.split(1)
            ],
            dim=0,
        )
        self.register_buffer("_identity_params", identity_params)

    def net_forward(self, x_masked: torch.Tensor) -> torch.Tensor:
        """Forward pass of the inner neural network.

        This method should return a tensor of shape
        ``(n_batch, n_params, *config_shape)``, i.e. the same shape
        as the sample being transformed, but with an extra dimension
        immediately after the batch dimension that is the number of
        parameters per degree of freedom being transforemd.

        If using fully connected networks:

        >>> v_in = x_masked[:, self._condition_mask]
        >>> ...
        >>> params = torch.zeros(
                x_masked.shape[0],          # batch size
                self._transform.n_params,   # parameters
                *x_masked.shape[1:]         # configuration shape
            )
        >>> params.masked_scatter_(self._transform_mask, v_out)
        >>> return params
        """
        raise NotImplementedError

    def forward(
        self, x: torch.Tensor, log_prob: torch.Tensor
    ) -> tuple[torch.Tensor]:
        params = self.net_forward(x.mul(self._condition_mask))
        params = params.mul(self._transform_mask)
        params = params.add(self._identity_params.mul(~self._transform_mask))
        """params = (
            self.net_forward(x.mul(self._condition_mask))
            .mul(self._transform_mask)
            .add(self._identity_params.mul(~self._transform_mask))
        )"""
        y, log_det_jacob = self._transform(x, params)
        log_prob.sub_(log_det_jacob.flatten(start_dim=1).sum(dim=1))
        return y, log_prob

    def inverse(
        self, y: torch.Tensor, log_prob: torch.Tensor
    ) -> tuple[torch.Tensor]:
        params = (
            self.net_forward(y.mul(self._condition_mask))
            .mul(self._transform_mask)
            .add(self._identity_params.mul(~self._transform_mask))
        )
        x, log_det_jacob = self._transform.inv(y, params)
        log_prob.sub_(log_det_jacob.flatten(start_dim=1).sum(dim=1))
        return x, log_det_jacob
