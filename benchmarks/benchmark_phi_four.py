import itertools

import pytest
import pytorch_lightning as pl
import torch

from torchnf.conditioners import MaskedConditioner, SimpleConditioner
from torchnf.layers import FlowLayer, Composition
from torchnf.models import BoltzmannGenerator, OptimizerConfig
from torchnf.networks import DenseNet, ConvNetCircular
from torchnf.transformers import Rescaling, RQSplineTransform
from torchnf.utils.distribution import diagonal_gaussian, IterableDistribution

from torchlft.phi_four.actions import PhiFourAction
from torchlft.phi_four.flows import EquivariantAffineTransform
from torchlft.utils import alternating_checkerboard_mask

L = 6  # lattice length
BETA = 0.537
LAM = 0.5

N_TRAIN = 2000
N_BATCH = 1000
N_VAL = 10


@pytest.fixture
def diagonal_gaussian_prior():
    return diagonal_gaussian([L, L])


@pytest.fixture
def phi_four_target():
    return PhiFourAction(beta=BETA, lamda=LAM)


@pytest.fixture
def checkerboard_mask():
    return alternating_checkerboard_mask([L, L])


@pytest.fixture
def optimizer():
    return OptimizerConfig(
        "Adam",
        {"lr": 0.001},
        "CosineAnnealingLR",
        {"T_max": N_TRAIN},
    )


@pytest.fixture
def trainer():
    return pl.Trainer(
        enable_checkpointing=False,
        # profiler="simple",
        max_epochs=1,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=2,  # turn off automatic validation
        # gpus=torch.cuda.device_count(),
    )


@pytest.fixture
def affine_flow_densenet(checkerboard_mask):
    depth = 8
    net = DenseNet(
        hidden_shape=[72],
        activation="Tanh",
        linear_kwargs=dict(bias=False),
    )
    layers = [
        FlowLayer(
            EquivariantAffineTransform(),
            MaskedConditioner(
                net(L**2 // 2, L**2), mask, mask_mode="index"
            ),
        )
        for _, mask in zip(range(depth), checkerboard_mask)
    ]
    layers.append(FlowLayer(Rescaling(), SimpleConditioner([0])))
    return Composition(*layers)


@pytest.fixture
def affine_flow_convnet(checkerboard_mask):
    depth = 8
    net = ConvNetCircular(
        dim=2,
        hidden_shape=[2, 2],
        activation="Tanh",
        kernel_size=3,
        conv_kwargs=dict(bias=False),
    )
    layers = [
        FlowLayer(
            EquivariantAffineTransform(),
            MaskedConditioner(
                net(1, 2), mask, mask_mode="mul", create_channel_dim=True
            ),
        )
        for _, mask in zip(range(depth), checkerboard_mask)
    ]
    layers.append(FlowLayer(Rescaling(), SimpleConditioner([0])))
    return Composition(*layers)


@pytest.fixture
def spline_flow_densenet(checkerboard_mask):
    depth = 2
    half_lattice = L**2 // 2
    net = DenseNet(
        hidden_shape=[72],
        activation="Tanh",
        skip_final_activation=True,
    )
    transformers = [
        RQSplineTransform(n_segments=8, interval=[-5, 5]) for _ in range(depth)
    ]
    conditioners = [
        MaskedConditioner(
            net(half_lattice, half_lattice * transformer.n_params),
            mask,
            mask_mode="index",
        )
        for transformer, mask in zip(transformers, checkerboard_mask)
    ]
    layers = [
        FlowLayer(transformer, conditioner)
        for transformer, conditioner in zip(transformers, conditioners)
    ]
    layers.append(FlowLayer(Rescaling(), SimpleConditioner([0])))
    return Composition(*layers)


def benchmark_affine_flow_densenet(
    diagonal_gaussian_prior,
    phi_four_target,
    affine_flow_densenet,
    optimizer,
    trainer,
):
    model = BoltzmannGenerator(
        affine_flow_densenet, diagonal_gaussian_prior, phi_four_target
    )
    model.configure_training(N_BATCH, N_TRAIN, val_epoch_length=N_VAL)

    optimizer.add_to(model)
    trainer.fit(model)
    (metrics,) = trainer.validate(model)

    assert metrics["Validation"]["AcceptanceRate"] > 0.4


def benchmark_affine_flow_convnet(
    diagonal_gaussian_prior,
    phi_four_target,
    affine_flow_convnet,
    optimizer,
    trainer,
):
    model = BoltzmannGenerator(
        affine_flow_convnet, diagonal_gaussian_prior, phi_four_target
    )
    model.configure_training(N_BATCH, N_TRAIN, val_epoch_length=N_VAL)

    optimizer.add_to(model)
    trainer.fit(model)
    (metrics,) = trainer.validate(model)

    assert metrics["Validation"]["AcceptanceRate"] > 0.4


def benchmark_spline_flow_densenet(
    diagonal_gaussian_prior,
    phi_four_target,
    spline_flow_densenet,
    optimizer,
    trainer,
):
    model = BoltzmannGenerator(
        spline_flow_densenet, diagonal_gaussian_prior, phi_four_target
    )
    model.configure_training(N_BATCH, N_TRAIN, val_epoch_length=N_VAL)

    optimizer.add_to(model)
    trainer.fit(model)
    (metrics,) = trainer.validate(model)

    assert metrics["Validation"]["AcceptanceRate"] > 0.65
