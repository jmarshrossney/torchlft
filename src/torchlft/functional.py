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

import logging

import torch

log = logging.getLogger(__name__)


def translation(x: torch.Tensor, shift: torch.Tensor) -> tuple[torch.Tensor]:
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
    :py:func:`torchlft.functional.transforms.inv_translation`
    """
    return x.add(shift), torch.zeros_like(x)


def inv_translation(
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


def affine_transform(
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
    :py:func:`torchlft.functional.transforms.inv_affine_transform`
    """
    return (x.mul(log_scale.neg().exp()).add(shift), log_scale.neg())


def inv_affine_transform(
    y: torch.Tensor, log_scale: torch.Tensor, shift: torch.Tensor
) -> tuple[torch.Tensor]:
    r"""Performs a pointwise affine transformation of the input tensor.

    .. math::

        y \mapsto x = (y - t) \odot e^{s}

        \log \left\lvert \frac{\partial x}{\partial y} \right\rvert = s

    See Also
    --------
    :py:func:`torchlft.functional.transforms.affine_transform`
    """
    return (y.sub(shift).mul(log_scale.exp()), log_scale)


def rq_spline_transform(
    x: torch.Tensor,
    widths: torch.Tensor,
    heights: torch.Tensor,
    derivs: torch.Tensor,
    knots_xcoords: torch.Tensor,
    knots_ycoords: torch.Tensor,
) -> torch.Tensor:

    outside_interval_mask = torch.logical_or(
        x < knots_xcoords[..., 0],
        x > knots_xcoords[..., -1],
    )
    if outside_interval_mask.sum() > 0.001 * x.numel():
        log.debug("More than 1/1000 inputs fell outside the spline interval")

    segment_idx = torch.searchsorted(knots_xcoords, x) - 1
    segment_idx.clamp_(0, widths.shape[-1])

    # Get parameters of the segments that x falls in
    w = torch.gather(widths, -1, segment_idx)
    h = torch.gather(heights, -1, segment_idx)
    d0 = torch.gather(derivs, -1, segment_idx)
    d1 = torch.gather(derivs, -1, segment_idx + 1)
    x0 = torch.gather(knots_xcoords, -1, segment_idx)
    y0 = torch.gather(knots_ycoords, -1, segment_idx)

    # NOTE: these will fail because some x are outside interval
    # Hence, alpha.clamp_(0, 1) will silently hide bugs
    # TODO: thnk of a smart and cheap way to check
    # eps = 1e-5
    # assert torch.all(x > x0 - eps)
    # assert torch.all(x < x0 + w + eps)

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

    y[outside_interval_mask] = x[outside_interval_mask]

    return y, gradient.log()


def inv_rq_spline_transform(
    y: torch.Tensor,
    widths: torch.Tensor,
    heights: torch.Tensor,
    derivs: torch.Tensor,
    knots_xcoords: torch.Tensor,
    knots_ycoords: torch.Tensor,
) -> torch.Tensor:

    outside_interval_mask = torch.logical_or(
        y < knots_ycoords[..., 0],
        y > knots_ycoords[..., -1],
    )
    if outside_interval_mask.sum() > 0.001 * y.numel():
        log.debug("More than 1/1000 inputs fell outside the spline interval")

    segment_idx = torch.searchsorted(knots_ycoords, y) - 1
    segment_idx.clamp_(0, widths.shape[-1])

    # Get parameters of the segments that x falls in
    w = torch.gather(widths, -1, segment_idx)
    h = torch.gather(heights, -1, segment_idx)
    d0 = torch.gather(derivs, -1, segment_idx)
    d1 = torch.gather(derivs, -1, segment_idx + 1)
    x0 = torch.gather(knots_xcoords, -1, segment_idx)
    y0 = torch.gather(knots_ycoords, -1, segment_idx)

    # eps = 1e-5
    # assert torch.all(y > y0 - eps)
    # assert torch.all(y < y0 + h + eps)

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
