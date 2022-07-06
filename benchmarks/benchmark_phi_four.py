import itertools

import pytest
import pytorch_lightning as pl
import torch

from torchnf.conditioners import MaskedConditioner, SimpleConditioner
from torchnf.flow import FlowLayer, Composition
from torchnf.models import BoltzmannGenerator, OptimizerConfig
from torchnf.networks import DenseNet, ConvNetCircular

# from torchnf.transformers import AffineTransform
from torchnf.utils.distribution import diagonal_gaussian, IterableDistribution

from torchlft.phi_four.actions import PhiFourActionIsing
from torchlft.phi_four.flows import AffineTransform
from torchlft.common.utils import make_checkerboard

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
    return PhiFourActionIsing(beta=BETA, lam=LAM)


@pytest.fixture
def checkerboard_mask():
    checkerboard = make_checkerboard([L, L])
    return itertools.cycle([checkerboard, ~checkerboard])


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


# temp fix while torchnf SimpleConditioner is broken
class GlobalRescaling(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.log_scale = torch.nn.Parameter(torch.tensor([0.0]))

    def forward(self, x, context):
        y = x.mul(self.log_scale.abs().neg().exp())
        ldj = (
            torch.ones_like(y)
            .mul(self.log_scale.abs().neg())
            .flatten(start_dim=1)
            .sum(dim=1)
        )
        return y, ldj


@pytest.fixture
def affine_flow_densenet(checkerboard_mask):
    depth = 8
    net = DenseNet(
        hidden_shape=[72],
        activation="Tanh",
        # skip_final_activation=True,
        linear_kwargs=dict(bias=False),
    )
    layers = [
        FlowLayer(
            AffineTransform(),
            MaskedConditioner(
                net(L ** 2 // 2, L ** 2), mask, mask_mode="index"
            ),
        )
        for _, mask in zip(range(depth), checkerboard_mask)
    ]
    # layers.append(FlowLayer(Rescaling(), SimpleConditioner([0])))
    layers.append(GlobalRescaling())
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
            AffineTransform(),
            MaskedConditioner(
                net(1, 2), mask, mask_mode="mul", create_channel_dim=True
            ),
        )
        for _, mask in zip(range(depth), checkerboard_mask)
    ]
    layers.append(GlobalRescaling())
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
