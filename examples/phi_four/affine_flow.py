import torch

from torchnf.model import FlowBasedModel

from torchlft.phi_four.flows import AffineCouplingFlow
from torchlft.phi_four.distributions import (
    SimpleGaussianPrior,
    PhiFourTargetIsing,
)
from torchlft.phi_four.observables import TwoPointObservables


class FlowBasedSampler(BoltzmannGenerator):
    def configure_optimizers(self) -> None:
        optimizer = torch.optim.Adam(self.flow.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.train_duration,
            eta_min=1e-6,
        )
        return optimizer, scheduler

    def fit(self, n_steps, *args, **kwargs):
        self.train_duration = n_steps
        super().fit(n_steps, *args, **kwargs)


def main():

    lattice_shape = (6, 6)
    batch_size = 1000
    beta = 0.537
    lam = 0.5
    n_train = 2000
    n_blocks = 6
    use_convnet = False
    hidden_shape = (72,)

    flow = AffineCouplingFlow(
        lattice_shape, n_blocks, use_convnet, hidden_shape
    )

    model = FlowBasedSampler(
        prior=SimpleGaussianPrior(lattice_shape, batch_size),
        target=PhiFourTargetIsing(beta, lam),
        flow=flow(),
    )
    model.fit(n_train, val_interval=n_train + 1, ckpt_interval=n_train + 1)
    metrics = model.validate()
    print(metrics["acceptance"])

    phi = model.mcmc_sample(10)

    torch.save(phi, "configs.pt")


def obs():
    phi = torch.load("configs.pt")
    tpo = TwoPointObservables(phi)

    print(tpo.correlator)
    print(tpo.low_momentum_correlation_length)


if __name__ == "__main__":
    obs()
