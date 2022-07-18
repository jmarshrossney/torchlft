import torch
import matplotlib.pyplot as plt

from torchlft.phi_four.actions import phi_four_action
from torchlft.phi_four.sample import HamiltonianMonteCarlo
from torchlft.sample.sampler import Sampler
from torchlft.sample.utils import autocorrelation


class Model(HamiltonianMonteCarlo):
    def init(self) -> None:
        super().init()
        self.magnetisation_history = []

    def on_sample(self):
        mag = self.state.mean()
        self.magnetisation_history.append(mag)
        self.log("magnetisation", mag)

    def on_final_sample(self):
        self.log("magnetisation_histogram", self.magnetisation_history)

        fig, ax = plt.subplots()
        ax.plot(autocorrelation(self.magnetisation_history))
        self.log("magnetisation_autocorr", fig)


def main():
    lattice_shape = [6, 6]
    beta = 0.537
    lamda = 0.5
    trajectory_length = 1.0
    steps = 20

    algorithm = Model(
        lattice_shape, trajectory_length, steps, beta=beta, lamda=lamda
    )

    sampler = Sampler(algorithm, "outputs/hmc")

    sampler.thermalise(1000)

    _ = sampler.sample(1000, 1)

    sampler.init()

    sampler.thermalise(1000)

    _ = sampler.sample(1000, 1)

    sampler._logger.close()


if __name__ == "__main__":
    main()
