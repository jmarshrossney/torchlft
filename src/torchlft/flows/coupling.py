from __future__ import annotations

from typing import Optional

import torch

import torchlft.transforms


class CouplingLayer(torch.nn.Module):
    def __init__(
        self,
        transform: torchlft.transforms.Transform,
        net: torch.nn.Module,
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

        self._transform = transform
        self._net = net
        self.register_buffer("_transform_mask", transform_mask)
        self.register_buffer("_condition_mask", condition_mask)

        self.register_buffer(
            "_transform_params",
            transform.get_identity_params(
                data_shape=condition_mask.unsqueeze(dim=0).shape
            ),
        )

    def _update_transform_params_buffer(
        self, data_shape: torch.Tensor
    ) -> None:
        if data_shape != self._transform_params.shape:
            self._transform_params = self._transform.get_identity_params(
                data_shape
            )

    def update_transform_params(self, x: torch.Tensor) -> None:
        net_inputs = x[:, self._condition_mask]
        net_outputs = self._net(net_inputs)
        self._transform_params.masked_scatter(
            self._transform_mask, net_outputs
        )

    def forward(
        self, x: torch.Tensor, log_det_jacob: torch.Tensor
    ) -> tuple[torch.Tensor]:
        self._update_transform_params_buffer(x.shape)
        self.update_transform_params(x)
        y, ldj = self._transform(x, self._transform_params)
        log_det_jacob.add_(ldj.flatten(start_dim=1).sum(dim=1))
        return y, log_det_jacob

    def inverse(
        self, y: torch.Tensor, log_det_jacob: torch.Tensor
    ) -> tuple[torch.Tensor]:
        self._update_transform_params_buffer(y.shape)
        self.update_transform_params(y)
        x, ldj = self._transform.inv(y, self._transform_params)
        log_det_jacob.add_(ldj.flatten(start_dim=1).sum(dim=1))
        return x, log_det_jacob


class CouplingLayerGeometryPreserving(torch.nn.Module):
    def update_transform_params(self, x: torch.Tensor) -> None:
        net_inputs = x.mul(self._condition_mask)
        net_outputs = self._net(net_inputs)
        net_outputs.mul_(self._transform_mask)
        self._transform_params.mul_(~self._transform_mask).add_(net_outputs)
