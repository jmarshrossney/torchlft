r"""
Module containing parametrised bijective transformations and their inverses.

Transformation functions take an input tensor and one or more tensors that
parametrise the transformation, and return the transformed inputs along
with the logarithm of the Jacobian determinant of the transformation. They
should be called as

    output, log_det_jacob = transform(input, param1, param2, ...)

In maths, the transformations do

.. math::

    x, \{\lambda\} \longrightarrow f(x ; \{\lambda\}),
    \log \left\lvert\frac{\partial f(x ; \{\lambda\})}{\partial x} \right\rvert

Note that the log-det-Jacobian is that of the *forward* transformation.
"""
from __future__ import annotations

import torch


def translation(x: torch.Tensor, shift: torch.Tensor, *args) -> tuple[torch.Tensor]:
    r"""Performs a pointwise translation of the input tensor.

    .. math::

        x \mapsto y = x + t

        \log \left\lvert \frac{\partial y}{\partial x} \right\rvert = 0

    Parameters
    ----------
    x
        Tensor to be transformed
    shift
        The translation, :math:`t`

    See Also
    --------
    :py:func:`torchlft.functional.transforms.translation_inv`
    """
    return x.add(shift), torch.zeros_like(x)


def translation_inv(
    y: torch.Tensor, shift: torch.Tensor
) -> tuple[torch.Tensor]:
    r"""Performs a pointwise translation of the input tensor.

    The inverse of :py:func:`translation`.

    .. math::

        y \mapsto x = y - t

        \log \left\lvert \frac{\partial x}{\partial y} \right\rvert = 0

    See Also
    --------
    :py:func:`torchlft.functional.transforms.translation`
    """
    return y.sub(shift), torch.zeros_like(y)


def affine(
    x: torch.Tensor, log_scale: torch.Tensor, shift: torch.Tensor
) -> tuple[torch.Tensor]:
    r"""Performs a pointwise affine transformation of the input tensor.

    .. math::

        x \mapsto y = x \odot e^{-s} + t

        \log \left\lvert \frac{\partial y}{\partial x} \right\rvert = -s

    Parameters
    ----------
    x
        Tensor to be transformed
    log_scale
        The scaling factor, :math:`s`
    shift
        The translation, :math:`t`

    See Also
    --------
    :py:func:`torchlft.functional.transforms.affine_inv`
    """
    return (x.mul(log_scale.neg().exp()).add(shift), log_scale.neg())


def affine_inv(
    y: torch.Tensor, log_scale: torch.Tensor, shift: torch.Tensor
) -> tuple[torch.Tensor]:
    r"""Performs a pointwise affine transformation of the input tensor.

    .. math::

        y \mapsto x = (y - t) \odot e^{s}

        \log \left\lvert \frac{\partial x}{\partial y} \right\rvert = s

    See Also
    --------
    :py:func:`torchlft.functional.transforms.affine`
    """
    return (y.sub(shift).mul(log_scale.exp()), log_scale)


def rational_quadratic_spline(
    x: torch.Tensor,
    segment_width: torch.Tensor,
    segment_height: torch.Tensor,
    lower_knot_deriv: torch.Tensor,
    upper_knot_deriv: torch.Tensor,
    lower_knot_x: torch.Tensor,
    lower_knot_y: torch.Tensor,
) -> torch.Tensor:
    eps = 1e-5
    assert torch.all(x > lower_knot_x - eps)
    assert torch.all(x < lower_knot_x + segment_width + eps)

    w, h, d0, d1, x0, y0 = (
        segment_width,
        segment_height,
        lower_knot_deriv,
        upper_knot_deriv,
        lower_knot_x,
        lower_knot_y,
    )
    s = h / w
    alpha = (x - x0) / w
    alpha.clamp_(0, 1)

    denominator_recip = torch.reciprocal(
        s + (d1 + d0 - 2 * s) * alpha * (1 - alpha)
    )
    beta = (s * alpha.pow(2) + d0 * alpha * (1 - alpha)) * denominator_recip
    y = y0 + h * beta

    gradient = (
        s.pow(2)
        * (
            d1 * alpha.pow(2)
            + 2 * s * alpha * (1 - alpha)
            + d0 * (1 - alpha).pow(2)
        )
        * denominator_recip.pow(2)
    )
    assert torch.all(gradient > 0)

    return y, gradient.log()


def rational_quadratic_spline_inv(
    y: torch.Tensor,
    segment_width: torch.Tensor,
    segment_height: torch.Tensor,
    lower_knot_deriv: torch.Tensor,
    upper_knot_deriv: torch.Tensor,
    lower_knot_x: torch.Tensor,
    lower_knot_y: torch.Tensor,
) -> torch.Tensor:
    eps = 1e-5
    assert torch.all(y > lower_knot_y - eps)
    assert torch.all(y < lower_knot_y + segment_width + eps)

    w, h, d0, d1, x0, y0 = (
        segment_width,
        segment_height,
        lower_knot_deriv,
        upper_knot_deriv,
        lower_knot_x,
        lower_knot_y,
    )
    s = h / w
    beta = (y - y0) / w
    beta.clamp_(0, 1)

    b = d0 - (d1 + d0 - 2 * s) * beta
    a = s - b
    c = -s * beta
    alpha = -2 * c * torch.reciprocal(b + (b.pow(2) - 4 * a * c).sqrt())
    x = x0 + w * alpha

    denominator_recip = torch.reciprocal(
        s + (d1 + d0 - 2 * s) * alpha * (1 - alpha)
    )
    gradient_fwd = (
        s.pow(2)
        * (
            d1 * alpha.pow(2)
            + 2 * s * alpha * (1 - alpha)
            + d0 * (1 - alpha).pow(2)
        )
        * denominator_recip.pow(2)
    )
    return x, gradient_fwd.log().neg()
