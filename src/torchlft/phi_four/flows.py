import dataclasses

import torch

import torchnf.recipes.networks
import torchnf.flow
import torchnf.transformers

import torchlft.utils


class AffineTransform(torchnf.transformers.AffineTransform):
    r"""
    Performs a pointwise affine transformation of the input tensor.

    This class modifies :py:class:`torchnf.transformers.AffineTransform`
    such that the log-scale parameter is replaced by its absolute value.
    This makes the transformation equivariant under global :math:`Z_2`
    symmetry (:math:`\phi \leftrightarrow -\phi`), provided the
    conditioner is also an equivariant function.

    The forward and inverse transformations are, respectively,

    .. math::

        x \mapsto y = x \odot e^{-|s|} + t

        y \mapsto x = (y - t) \odot e^{|s|}

    .. math::

        \log \left\lvert \frac{\partial y}{\partial x} \right\rvert
        = \sum_i -|s_i|

    where :math:`i` runs over the degrees of freedom being transformed.
    """

    def _forward(
        self, x: torch.Tensor, params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_scale, shift = params.split(1, dim=1)
        params = torch.cat([log_scale.abs(), shift], dim=1)
        return super()._forward(x, params)


@dataclasses.dataclass
class AffineCouplingFlow:
    lattice_shape: tuple[int, int]
    n_blocks: int
    use_convnet: bool
    hidden_shape: tuple[int]
    activation: str = "Tanh"  # or Tanhshrink

    @property
    def densenet(self) -> torchnf.recipes.networks.DenseNet:
        lattice_sites = self.lattice_shape[0] * self.lattice_shape[1]
        return torchnf.recipes.networks.DenseNet(
            in_features=lattice_sites // 2,
            out_features=lattice_sites,  # 2 params (s, t) for each site
            hidden_shape=self.hidden_shape,
            activation=self.activation,
            skip_final_activation=True,
            linear_kwargs={"bias": False},  # equivariance
        )

    @property
    def convnet(self) -> torchnf.recipes.networks.ConvNet:
        return torchnf.recipes.networks.ConvNetCircular(
            dim=2,
            in_channels=1,
            out_channels=2,
            hidden_shape=self.hidden_shape,
            activation=self.activation,
            skip_final_activation=True,
            conv_kwargs={"bias": False},
        )

    @property
    def mask(self) -> torch.BoolTensor:
        return torchlft.utils.make_checkerboard(self.lattice_shape)

    def __call__(self) -> torchnf.flow.Flow:
        net = self.convnet if self.use_convnet else self.densenet
        mask = self.mask

        layers = []
        for _ in range(self.n_blocks):
            layers.append(
                torchnf.flow.FlowLayer(
                    AffineTransform(),
                    torchnf.conditioners.MaskedConditioner(net(), mask),
                )
            )
            layers.append(
                torchnf.flow.FlowLayer(
                    AffineTransform(),
                    torchnf.conditioners.MaskedConditioner(net(), ~mask),
                )
            )
        return torchnf.flow.Flow(*layers)
