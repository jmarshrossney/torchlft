import torch
import matplotlib.pyplot as plt

from torchlft.phi_four.actions import phi_four_action
from torchlft.phi_four.sample import RandomWalkMetropolis
from torchlft.sample.sampler import Sampler
from torchlft.sample.utils import autocorrelation


class Model(RandomWalkMetropolis):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mag = []
        self.action = []

    def on_sample(self):
        mag = self.state.mean()
        action = phi_four_action(
            self.state.unsqueeze(0), **self.couplings
        ).squeeze()
        self.mag.append(mag)
        self.action.append(action)
        self.log("mag", mag)
        self.log("action", action)

    def on_final_sample(self):
        self.mag_corr = autocorrelation(self.mag)
        self.action_corr = autocorrelation(self.action)

        self.log("mag_hist", self.mag)
        self.log("action_hist", self.action)

        fig, ax = plt.subplots()
        ax.plot(self.mag_corr)
        self.log("corr/mag", fig)

        fig, ax = plt.subplots()
        ax.plot(self.action_corr)
        self.log("corr/action", fig)


def main():
    lattice_shape = [6, 6]
    beta = 0.537
    lamda = 0.5
    step_size = 0.5

    algorithm = Model(lattice_shape, step_size, beta=beta, lamda=lamda)

    sampler = Sampler(algorithm, "outputs/metropolis")

    sampler.thermalise(5000)

    _ = sampler.sample(1000, 10)

    # sampler.init()

    # sampler.thermalise(5000)

    # _ = sampler.sample(1000, 10)


if __name__ == "__main__":
    main()
