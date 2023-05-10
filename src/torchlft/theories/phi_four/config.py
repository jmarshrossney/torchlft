from torchlft.typing import *


class ScalarFieldConfig(FieldConfig):
    def __init__(self, data: Tensor, lattice_shape: torch.Size) -> None:
        batch_size, *lattice_dims = data.shape

        if math.prod(lattice_dims) != math.prod(lattice_shape):
            ...
        self._batch_size = batch_size
