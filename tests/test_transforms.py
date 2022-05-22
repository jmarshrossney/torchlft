from hypothesis import given, strategies as st
import random
import torch

from torchlft.transforms import (
    Tanh,
    Translation,
    Rescaling,
    AffineTransform,
    RQSplineTransform,
    RQSplineTransformLinearTails,
    RQSplineTransformCircular,
)


def _test_call(transform, x, params):
    _, _ = transform(x, params)
    _, _ = transform.inv(x, params)


def _test_identity(transform, x):
    params = torch.stack(
        [param.expand_as(x) for param in transform.identity_params.split(1)],
        dim=transform.params_dim,
    )
    y, ldj = transform(x, params)
    assert torch.allclose(x, y)
    assert torch.allclose(ldj, torch.zeros_like(ldj))

    x, ldj = transform.inv(y, params)
    assert torch.allclose(y, x)
    assert torch.allclose(ldj, torch.zeros_like(ldj))


def _test_roundtrip(transform, x, params):
    y, ldj_fwd = transform(x, params)
    xx, ldj_inv = transform.inv(y, params)
    assert torch.allclose(x, xx, atol=1e-5)
    assert torch.allclose(ldj_fwd, ldj_inv.neg(), atol=1e-5)


def test_tanh():
    x = torch.empty(10, 10).normal_()
    y, ldj_fwd = Tanh()(x)
    xx, ldj_inv = Tanh().inv(y)
    assert torch.allclose(x, xx)
    assert torch.allclose(ldj_fwd, ldj_inv.neg())


@given(x_shape=st.lists(st.integers(1, 10), min_size=2, max_size=5))
def test_translation(x_shape):
    x = torch.empty(x_shape).normal_()
    shift = torch.empty(x_shape).normal_()

    _test_call(Translation(), x, shift)
    _test_identity(Translation(), x)
    _test_roundtrip(Translation(), x, shift)


@given(x_shape=st.lists(st.integers(1, 10), min_size=2, max_size=5))
def test_rescaling(x_shape):
    x = torch.empty(x_shape).normal_()
    log_scale = torch.empty(x_shape).normal_()

    _test_call(Rescaling(), x, log_scale)
    _test_identity(Rescaling(), x)
    _test_roundtrip(Rescaling(), x, log_scale)


@given(x_shape=st.lists(st.integers(1, 10), min_size=2, max_size=5))
def test_affine_transform(x_shape):
    x = torch.empty(x_shape).normal_()
    log_scale = torch.empty(x_shape).normal_()
    shift = torch.empty(x_shape).normal_()

    params_dim = random.randint(-1, len(x_shape))
    params = torch.stack([log_scale, shift], dim=params_dim)
    transform = AffineTransform(params_dim=params_dim)

    _test_call(transform, x, params)
    _test_identity(transform, x)
    _test_roundtrip(transform, x, params)
