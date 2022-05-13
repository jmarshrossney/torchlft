from hypothesis import given, strategies as st
import pytest
import torch

from torchlft.functional.transforms import (
    translation,
    inv_translation,
    rescaling,
    inv_rescaling,
    affine_transform,
    inv_affine_transform,
    rq_spline_transform,
    inv_rq_spline_transform,
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
    assert torch.allclose(inputs, outputs, atol=1e-5)
    assert torch.allclose(
        log_det_jacob_fwd, log_det_jacob_inv.neg(), atol=1e-5
    )


@given(input_shape=st.lists(st.integers(1, 10), min_size=1, max_size=5))
def test_translation(input_shape):
    x = torch.empty(input_shape).normal_()
    shift = torch.empty(input_shape).normal_()
    zeros = torch.zeros_like(shift)

    _test_call(translation, x, shift)
    _test_call(inv_translation, x, shift)

    _test_identity(translation, x, zeros)
    _test_identity(inv_translation, x, zeros)

    _test_notinplace(translation, x, shift)
    _test_notinplace(inv_translation, x, shift)

    _test_roundtrip(translation, inv_translation, x, shift)


@given(input_shape=st.lists(st.integers(1, 10), min_size=1, max_size=5))
def test_rescaling(input_shape):
    x = torch.empty(input_shape).normal_()
    log_scale = torch.empty(input_shape).normal_()
    zeros = torch.zeros_like(log_scale)

    _test_call(rescaling, x, log_scale)
    _test_call(inv_rescaling, x, log_scale)

    _test_identity(rescaling, x, zeros)
    _test_identity(inv_rescaling, x, zeros)

    _test_notinplace(rescaling, x, log_scale)
    _test_notinplace(inv_rescaling, x, log_scale)

    _test_roundtrip(rescaling, inv_rescaling, x, log_scale)


@given(input_shape=st.lists(st.integers(1, 10), min_size=1, max_size=5))
def test_affine_transform(input_shape):
    x = torch.empty(input_shape).normal_()
    log_scale = torch.empty(input_shape).normal_()
    shift = torch.empty(input_shape).normal_()
    zeros = torch.zeros_like(shift)

    _test_call(affine_transform, x, log_scale, shift)
    _test_call(inv_affine_transform, x, log_scale, shift)

    _test_identity(affine_transform, x, zeros, zeros)
    _test_identity(inv_affine_transform, x, zeros, zeros)

    _test_notinplace(affine_transform, x, log_scale, shift)
    _test_notinplace(inv_affine_transform, x, log_scale, shift)

    _test_roundtrip(
        affine_transform, inv_affine_transform, x, log_scale, shift
    )
