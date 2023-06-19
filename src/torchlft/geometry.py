from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn


class Geometry(nn.Module, metaclass=ABCMeta):
    @property
    @abstractmethod
    def dim(self) -> int:
        ...

    @abstractmethod
    def create_masks(self) -> tuple[BoolTensor, ...]:
        ...

    def _update_masks(self):
        masks = self.create_masks()

        assert [mask.dtype == torch.bool for mask in masks]

        # Check that partitions are disjoint and union = entire lattice
        assert torch.all(torch.stack(masks).sum(dim=0) == 1)

        list_of_masks = []
        for i, mask in enumerate(masks):
            # Overwrites existing buffer
            self.register_buffer(f"_mask_{i}", mask)
            list_of_masks.append(getattr(self, f"_mask_{i}"))

        self._masks = list_of_masks

    @property
    def lattice_shape(self) -> tuple[int, ...]:
        return self._lattice_shape

    @lattice_shape.setter
    def lattice_shape(self, new_shape: tuple[int, ...]) -> None:
        assert len(new) == self.dim
        assert all([length > 0 for length in new_shape])
        assert all([length % 2 == 0 for length in new_shape])
        self._lattice_shape = lattice_shape
        self._update_masks()

    def get_mask(self, idx: int) -> BoolTensor:
        return self._masks[idx]

    def partition_as_lexi(self, inputs: Tensor) -> tuple[Tensor, ...]:
        return tuple([inputs[:, mask] for mask in self._list_of_masks])

    def restore_from_lexi(self, *partitions: tuple[Tensor, ...]) -> Tensor:
        first = partitions[0]
        batch_size, *extra_dims = first.shape
        outputs = first.new_empty(batch_size, *self.lattice_shape, *extra_dims)
        for mask, partition in zip(
            self._list_of_masks, partitions, strict=True
        ):
            outputs[:, mask] = partition
        return outputs

    def partition_as_masked(self, inputs: Tensor) -> tuple[Tensor, ...]:
        return tuple(
            [
                inputs.masked_fill(mask, float("NaN"))
                for mask in self._list_of_masks
            ]
        )

    def restore_from_masked(self, *partitions: tuple[Tensor, ...]) -> Tensor:
        return sum(partitions.nan_to_num())

    def lexi_as_masked(self, partition: Tensor, idx: int) -> Tensor:
        batch_size, *extra_dims = partition.shape
        output = partition.new_full(
            size=(batch_size, *self.lattice_shape, *extra_dims),
            fill_value=float("NaN"),
        )
        output[:, self.get_mask(idx)] = partition
        return output

    def masked_as_lexi(self, partition: Tensor, idx: int) -> Tensor:
        return partition[:, self.get_mask(idx)]


class CheckerboardGeometry1D(Geometry):
    dim = 1

    def create_masks(self) -> tuple[BoolTensor, BoolTensor]:
        checkerboard = torch.zeros(self.lattice_shape, dtype=torch.bool)
        checkerboard[::2] = True
        return checkerboard, ~checkerboard


class CheckerboardGeometry2D(Geometry):
    dim = 2

    def create_masks(self) -> tuple[BoolTensor, BoolTensor]:
        checkerboard = torch.zeros(self.lattice_shape, dtype=torch.bool)
        checkerboard[::2, ::2] = True
        checkerboard[1::2, 1::2] = True
        return checkerboard, ~checkerboard


# TODO: partitioning of gauge links

class CouplingLayer(nn.Module):
    def __init__(self, transform: Callable[Tensor, Transform], net_factory: NetFactory, geometry: Geometry):
        super().__init__()

        ...

    def conditional_structure(self) -> list[int, tuple[int, ...]]:
        ...

    def prepare_net_inputs(self, *partitions: Tensor):
        ...

    def forward(self, inputs):
        partitions = self.geometry.partition_as_masked(inputs)

        for (active_idx, passive_indices), net in zip(self.conditional_structure, self.networks):
            active_partition = partitions[active_idx]
            passive_partitions = [partitions[ip] for ip in passive_indices]
            net_inputs = self.prepare_net_inputs(passive_partitions)
            transform_params = net(net_inputs)
            f = self.transform(transform_params)
            ϕ_active = f(ϕ_active)
            partitions




