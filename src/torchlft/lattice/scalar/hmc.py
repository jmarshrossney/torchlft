from math import isclose
from typing import TypeAlias

import torch
import torch.nn as nn

from torchlft.lattice.sample import SamplingAlgorithm
from torchlft.lattice.scalar.action import Phi4Action
from torchlft.utils.torch import dict_concat

Tensor: TypeAlias = torch.Tensor


# Separable Hamiltonian
class Hamiltonian(nn.Module):
    def __init__(self, action):
        super().__init__()
        self.register_module("action", action)

    def forward(self, x: Tensor, p: Tensor) -> Tensor:
        return 0.5 * p.pow(2).flatten(1).sum(1, keepdim=True) + self.action(x)

    def grad_wrt_coords(self, x: Tensor) -> Tensor:
        return self.action.grad(x)

    def grad_wrt_momenta(self, p: Tensor) -> Tensor:
        return p.clone()

    def sample_momenta(
        self, x0: Tensor, generator: torch.Generator | None = None
    ) -> Tensor:
        p0 = torch.empty_like(x0).normal_(generator=generator)
        return p0


class LeapfrogIntegrator:
    def __init__(
        self,
        hamiltonian,
        step_size: float,
        traj_length: float = 1.0,
    ):

        # TODO: prioritise traj_length or step_size?
        n_steps = max(1, round(traj_length / abs(step_size)))
        real_traj_length = n_steps * step_size
        if not isclose(real_traj_length, traj_length):
            print("baddies")  # TODO

        self.hamiltonian = hamiltonian
        self.step_size = step_size
        self.n_steps = n_steps
        self.traj_length = traj_length

    def on_step(self, x: Tensor, p: Tensor, t: float) -> None:
        pass

    def _integrate(self, x, p, t, inverse):

        x = x.clone()
        p = p.clone()
        ε = -self.step_size if inverse else self.step_size

        F = self.hamiltonian.grad_wrt_coords(x).negative()

        for _ in range(self.n_steps):
            self.on_step(x, p, t)

            # NOTE: avoid in-place here in case p stored in on_step_func
            p = p + (ε / 2) * F

            v = self.hamiltonian.grad_wrt_momenta(p)

            x = x + ε * v

            F = self.hamiltonian.grad_wrt_coords(x).negative()

            p += (ε / 2) * F

            t += ε

        return x, p, t

    def __call__(
        self, coords: Tensor, momenta: Tensor
    ) -> tuple[Tensor, Tensor, float]:
        return self._integrate(coords, momenta, t=0, inverse=False)

    def inverse(
        self, coords: Tensor, momenta: Tensor
    ) -> tuple[Tensor, Tensor, float]:
        return self._integrate(
            coords, momenta, t=self.traj_length, inverse=True
        )


class HybridMonteCarlo(SamplingAlgorithm):
    def __init__(
        self,
        lattice: tuple[int, ...],
        integrator: LeapfrogIntegrator,
        n_replica: int = 1,
    ):
        self.lattice = lattice
        self.n_replica = n_replica
        self.integrator = integrator

        self.hamiltonian = integrator.hamiltonian

        # NOTE: should I leave this unset to prevent 'update' being called before 'init'?
        self.rng = torch.Generator("cpu")

    def init(self, rng_seed: int | None = None):

        if rng_seed is not None:
            self.rng.manual_seed(rng_seed)

        state = torch.empty(self.n_replica, *self.lattice).normal_(
            0, 1, generator=self.rng
        )

        # flatten??

        return state.flatten(1)

    def update(self, state: Tensor) -> tuple[Tensor, dict[str, Tensor]]:

        φ0 = state
        ω0 = self.hamiltonian.sample_momenta(φ0, generator=self.rng)

        H0 = self.hamiltonian(φ0, ω0)

        φt, ωt, t = self.integrator(φ0, ω0)

        Ht = self.hamiltonian(φt, ωt)

        unif = torch.rand(Ht.shape, generator=self.rng)
        accepted = torch.exp(Ht - H0) > unif

        φt = torch.where(accepted, φt, φ0)

        log_data = {"ΔH": H0 - Ht, "accepted": accepted}

        return φt, log_data

    def compute_stats(self, log_data: list[dict[str, Tensor]]):
        # TODO: actually useful stats, mean plus error

        log_data = dict_concat(log_data)

        ΔH = log_data["ΔH"]
        stat = torch.exp(ΔH).mean() - 1  # TODO

        acceptance = log_data["accepted"]
        print("acc shape: ", acceptance.shape)
        acceptance = acceptance.float().mean()

        return {"zero_stat": stat, "acceptance": acceptance}


def main():
    from torchlft.lattice.sample import DefaultSampler

    sampler = DefaultSampler(1000, 100, 1234567)
    hamiltonian = Hamiltonian(Phi4Action(β=0.5, λ=0.5))
    integrator = LeapfrogIntegrator(hamiltonian, step_size=0.05)
    alg = HybridMonteCarlo([6, 6], integrator)
    configs, stats = sampler.sample(alg)

    print("configs shape: ", configs.shape)
    print(stats)


if __name__ == "__main__":
    main()
