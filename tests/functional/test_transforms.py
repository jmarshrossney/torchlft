from hypothesis import given, strategies as st
import pytest
import torch

from torchlft.functional.transforms import (
    translation,
    translation_inv,
    affine,
    affine_inv,
    rational_quadratic_spline,
    rational_quadratic_spline_inv,
)


def _test_call(transform, inputs, *params):
    _, _ = transform(inputs, *params)


def _test_notinplace(transform, inputs, *params):
    x = inputs.clone()
    _, _ = transform(inputs, *params)
    assert torch.allclose(x, inputs)


def _test_identity(transform, inputs, *params):
    outputs, log_det_jacob = transform(inputs, *params)
    assert torch.allclose(inputs, outputs)
    assert torch.allclose(log_det_jacob, torch.zeros_like(log_det_jacob))


def _test_roundtrip(transform, transform_inv, inputs, *params):
    intermediates, log_det_jacob_fwd = transform(inputs, *params)
    outputs, log_det_jacob_inv = transform_inv(intermediates, *params)
    assert torch.allclose(inputs, outputs, atol=1e-6)
    assert torch.allclose(
        log_det_jacob_fwd, log_det_jacob_inv.neg(), atol=1e-6
    )


@given(input_shape=st.lists(st.integers(1, 10), min_size=1, max_size=5))
def test_translation(input_shape):
    x = torch.empty(input_shape).normal_()
    shift = torch.empty(input_shape).normal_()
    zeros = torch.zeros_like(shift)

    _test_call(translation, x, shift)
    _test_call(translation_inv, x, shift)

    _test_identity(translation, x, zeros)
    _test_identity(translation_inv, x, zeros)

    _test_notinplace(translation, x, shift)
    _test_notinplace(translation_inv, x, shift)

    _test_roundtrip(translation, translation_inv, x, shift)


@given(input_shape=st.lists(st.integers(1, 10), min_size=1, max_size=5))
def test_affine(input_shape):
    x = torch.empty(input_shape).normal_()
    log_scale = torch.empty(input_shape).normal_()
    shift = torch.empty(input_shape).normal_()
    zeros = torch.zeros_like(shift)

    _test_call(affine, x, log_scale, shift)
    _test_call(affine_inv, x, log_scale, shift)

    _test_identity(affine, x, zeros, zeros)
    _test_identity(affine_inv, x, zeros, zeros)

    _test_notinplace(affine, x, log_scale, shift)
    _test_notinplace(affine_inv, x, log_scale, shift)

    _test_roundtrip(affine, affine_inv, x, log_scale, shift)

# TODO test rational quadratic spline
