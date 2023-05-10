import torch

import torchnf.transformers


class EquivariantAffineTransform(torchnf.transformers.AffineTransform):
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

    def _inverse(
        self, y: torch.Tensor, params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_scale, shift = params.split(1, dim=1)
        params = torch.cat([log_scale.abs(), shift], dim=1)
        return super()._inverse(y, params)
