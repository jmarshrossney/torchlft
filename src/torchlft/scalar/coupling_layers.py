from typing import TypeAlias

import torch
import torch.nn as nn

from torchlft.nflow.nn import Activation, ConvNet2d, PointNet
from torchlft.nflow.partition import Checkerboard2d
from torchlft.nflow.transforms.core import UnivariateTransformModule


Tensor: TypeAlias = torch.Tensor



class CouplingLayer(nn.Module):
    def __init__(
        self,
        transform: UnivariateTransformModule,
        spatial_net: nn.Module,  # conv
        layer_id: int,
    ):
        super().__init__()

        self.register_module("transform", transform)
        self.register_module("spatial_net", spatial_net)

        partitioning = Checkerboard2d(partition_id=layer_id)
        self.register_module("partitioning", partitioning)

    def forward(self, φ_in: Tensor) -> tuple[Tensor, Tensor]:
        # Get active and frozen masks
        _, L, T, _ = φ_in.shape
        active_mask, frozen_mask = self.partitioning(dimensions=(L, T))

        # Construct conditional transformation
        net_inputs = frozen_mask.unsqueeze(-1) * φ_in
        context = self.spatial_net(net_inputs)[:, active_mask]
        transform = self.transform(context)

        # Transform active variables
        φ_out = φ_in.clone()
        φ_out[:, active_mask], ldj = transform(φ_in[:, active_mask])

        return φ_out, ldj


class LegacyCouplingLayer(nn.Module):
    def __init__(
        self,
        transform: UnivariateTransformModule,
        net: nn.Module,
        layer_id: int,
    ):
        super().__init__()
        self.register_module("transform", transform)
        self.register_module("net", net)

        self.layer_id = layer_id


    def split(self, φ_in: Tensor) -> tuple[Tensor, Tensor]:
        φ_A, φ_B = φ_in.tensor_split(2, dim=1)
        if self.layer_id % 2:
            return φ_B, φ_A
        else:
            return φ_A, φ_B

    def join(self, φ_a: Tensor, φ_p: Tensor) -> tuple[Tensor, Tensor]:
        if self.layer_id % 2:
            return torch.cat([φ_p, φ_a], dim=1)
        else:
            return torch.cat([φ_a, φ_p], dim=1)
    
    def forward(self, φ_in: Tensor) -> tuple[Tensor, Tensor]:
        N, D = φ_in.shape
        assert D % 2 == 0

        φ_a, φ_p = self.split(φ_in)

        # Construct conditional transformation
        context = self.net(φ_p).view(*φ_a.shape, -1)
        transform = self.transform(context)

        # Transform active variables
        φ_a, ldj = transform(φ_a.unsqueeze(-1))

        φ_out = self.join(φ_a.squeeze(-1), φ_p)

        return φ_out, ldj
