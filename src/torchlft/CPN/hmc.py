from functools import partial
from math import prod, pi as π
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.linalg as LA
from tqdm import trange

import matplotlib.pyplot as plt

from torchlft.CPN.action import action_v2, action_v3, grad_action_v3
from torchlft.CPN.observables import *
from torchlft.observables import TwoPointObservables, IntegratedAutocorrelation
from torchlft.sample import HybridMonteCarlo, Sampler, metropolis_test
from torchlft.utils.linalg import dot, outer, orthogonal_projection
from torchlft.utils.tensor import mod_2pi, sum_except_batch

Tensor = torch.Tensor
BoolTensor = torch.BoolTensor


class Hamiltonian(nn.Module):
    def __init__(self, g: float):
        super().__init__()
        self.g = g

    def kinetic(self, pω: tuple[Tensor, Tensor]) -> Tensor:
        p, ω = pω
        return sum_except_batch(dot(p, p) / 2) + sum_except_batch(ω**2 / 2)

    def potential(self, xA: tuple[Tensor, Tensor]) -> Tensor:
        x, A = xA
        return action_v3(x, A, self.g)

    def compute(
        self, xA: tuple[Tensor, Tensor], pω: tuple[Tensor, Tensor]
    ) -> Tensor:
        return self.kinetic(pω) + self.potential(xA)

    def grad_kinetic(self, pω: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        p, ω = pω
        return (p.clone(), ω.clone())

    def grad_potential(
        self, xA: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, Tensor]:
        x, A = xA
        return grad_action_v3(x, A, self.g)

    def sample_momenta(
        self, xA: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, Tensor]:
        x, A = xA
        p = torch.empty_like(x).normal_()
        p = orthogonal_projection(p, x)
        ω = torch.empty_like(A).normal_()
        return (p, ω)


def leapfrog(
    coords: tuple[Tensor, Tensor],
    momenta: tuple[Tensor, Tensor],
    hamiltonian: Hamiltonian,
    *,
    step_size: float,
    traj_length: float,
    on_step_func: Optional[Callable[[Tensor, Tensor, float], None]] = None,
) -> tuple[Tensor, Tensor, float]:
    n_steps = max(1, round(traj_length / abs(step_size)))

    x0, A0 = coords
    p0, ω0 = momenta
    x, A, p, ω = x0.clone(), A0.clone(), p0.clone(), ω0.clone()
    t = 0
    ε = step_size

    dHdx, dHdA = hamiltonian.grad_potential((x, A))

    for _ in range(n_steps):
        if on_step_func is not None:
            on_step_func((x, A), (p, ω), t)

        # Update momenta
        p = p - (ε / 2) * dHdx
        ω = ω - (ε / 2) * dHdA

        # Update coordinates
        dHdp, dHdω = hamiltonian.grad_kinetic((p, ω))

        # Geodesic update of (x, p)
        v = dHdp
        assert torch.allclose(
            dot(x, v), torch.zeros(1), atol=1e-5
        ), f"{dot(x, v).abs().max()}"
        mod_p = LA.vector_norm(p, dim=-1, keepdim=True)
        mod_v = LA.vector_norm(v, dim=-1, keepdim=True)
        cos_εv = (ε * mod_v).cos()
        sin_εv = (ε * mod_v).sin()

        p = cos_εv * p - (sin_εv * mod_p) * x
        x = cos_εv * x + (sin_εv / mod_v) * v

        mod_x = LA.vector_norm(x, dim=-1, keepdim=True)
        x /= mod_x

        # Update A
        A = mod_2pi(A + ε * dHdω)

        # Update momenta
        dHdx, dHdA = hamiltonian.grad_potential((x, A))

        p = p - (ε / 2) * dHdx
        ω = ω - (ε / 2) * dHdA

        t += ε

    return (x, A), (p, ω), t


class HMC(HybridMonteCarlo):
    def __init__(
        self,
        lattice_shape: tuple[int, int],
        N: int,
        g: float,
        step_size: float,
    ):
        self.lattice_shape = lattice_shape
        self.N = N

        hamiltonian = Hamiltonian(g)
        integrator = partial(
            leapfrog,
            step_size=step_size,
            traj_length=1,
        )
        super().__init__(hamiltonian, integrator)

    def init(self):
        x = torch.empty(1, *self.lattice_shape, 2 * self.N).normal_()
        x = x / LA.vector_norm(x, dim=-1, keepdim=True)
        A = torch.empty(1, *self.lattice_shape, 2).uniform_(0, 2 * π)

        del self.state

        self.state = (x, A)

    def forward(self) -> BoolTensor:
        x0 = self.state
        p0 = self.hamiltonian.sample_momenta(x0)
        H0 = self.hamiltonian.compute(x0, p0)

        xT, pT, T = self.integrator(
            coords=x0,
            momenta=p0,
            hamiltonian=self.hamiltonian,
        )

        HT = self.hamiltonian.compute(xT, pT)

        # print(H0, HT)

        self._delta_hamiltonian.append(H0 - HT)

        accepted = metropolis_test(
            current=-H0, proposal=-HT
        ).item()  # 1 at a time

        if accepted:
            self.state = xT

        return accepted


lattice = (42, 42)
N = 10
g = 1 / 7
ε = 1 / 62

#hmc = HMC(lattice, N, g, ε)
#hmc.init()

n = 10000
nb = 200
accepted = 0
x, A = [], []
"""

# Burn in
for _ in trange(nb):
    hmc()

for _ in trange(n):
    a = hmc()
    accepted += a
    x_, A_ = hmc.state
    x.append(x_)
    A.append(A_)

x = torch.cat(x)
A = torch.cat(A)
"""
#torch.save(x, "x.pt")
#torch.save(A, "A.pt")

x = torch.load("x.pt")
A = torch.load("A.pt")

z = torch.complex(x[..., ::2], x[..., 1::2])

#print("acceptance: ", accepted / n)
#print("exp(ΔH): ", hmc.exp_delta_hamiltonian())
"""
# Energy
energy = []
for xx, AA in zip(x.split(1000, 0), A.split(1000, 0)):
    energy.append(action_v3(xx, AA, g) * (g / prod(lattice)))

energy = torch.cat(energy)

fig, ax = plt.subplots()
ax.plot(range(nb, n + nb), energy)
ax.set_ylabel("energy")
ax.set_xlabel("MD time")
fig.savefig("energy.png")

# Energy autocorrelation
tau_energy = IntegratedAutocorrelation(energy)
tau = tau_energy.compute().item()
t = range(0, min(n, round(10 * tau)))
fig, ax = plt.subplots()
ax.plot(t, tau_energy.autocorrelation[t], label="autocorrelation")
ax.plot(t, tau_energy.autocorrelation.cumsum(dim=-1)[t], label="cumulative")
ax.axvline(
    tau, ls="--", color="red", label=r"$\tau_{int}$"
)
ax.set_xlabel("MD time difference")
ax.set_title("Energy density")
ax.legend()
fig.savefig("tau_energy.png")
"""
"""# Susceptibility
P = outer(z, z.conj()).flatten(start_dim=1, end_dim=2)
correlator = torch.einsum("bxij,byij->bxy", P, P).real
susc = torch.mean(correlator, dim=(1, 2))

# Susceptibility autocorrelation
tau_susc = IntegratedAutocorrelation(susc)
tau = tau_susc.compute().item()
t = range(0, min(n, round(10 * tau)))
fig, ax = plt.subplots()
ax.plot(t, tau_susc.autocorrelation[t], label="autocorrelation")
ax.plot(t, tau_susc.autocorrelation.cumsum(dim=-1)[t], label="cumulative")
ax.axvline(
    tau, ls="--", color="red", label=r"$\tau_{int}$"
)
ax.set_xlabel("MD time difference")
ax.set_title("Susceptibility")
ax.legend()
fig.savefig("tau_susc.png")
"""
# Two point observables
observables = TwoPointObservables(
    #z[::10],
    torch.stack(z.split(100, dim=0)),
    two_point_correlator,
    #n_bootstrap=100,
    contains_replicas=True
)

fig, ax = plt.subplots()
im = ax.imshow(observables.correlator.mean(dim=0))
fig.colorbar(im)
fig.savefig("correlator.png")

fig, ax = plt.subplots()
G = observables.zero_momentum_correlator
ax.errorbar(x=torch.arange(G.shape[1]), y=G.mean(0), yerr=G.std(0))
ax.set_xlabel("lattice x")
ax.set_ylabel("zero momentum correlator (log scale)")
ax.set_yscale("log")
fig.savefig("zero_mom.png")

fig, ax = plt.subplots()
mp = observables.effective_pole_mass
ax.errorbar(x=torch.arange(mp.shape[1]), y=mp.mean(0), yerr=mp.std(0))
ax.set_xlabel("lattice x")
ax.set_ylabel("effective pole mass (acosh)")
fig.savefig("pole_mass.png")

e, chi, xi = observables.energy_density, observables.susceptibility, observables.correlation_length
print(f"energy density: {e.mean()} +/- {e.std()}")
print(f"susceptibility: {chi.mean()} +/- {chi.std()}")
print(f"correlation_length: {xi.mean()} +/- {xi.std()}")

"""
# Topology
Q1, Q2, Q3 = [], [], []
for zz, AA in zip(z.split(100, 0), A.split(100, 0)):
    Q1.append(topological_charge_geometric(zz))
    Q2.append(topological_charge_v2(zz))
    Q3.append(topological_charge_v3(AA))

Q1 = torch.cat(Q1)
Q2 = torch.cat(Q2)
Q3 = torch.cat(Q3)

print("Q geom: ", Q1.mean())
print("Q angle: ", Q2.mean())
print("Q gauge: ", Q3.mean())
print("10^5 Q^2 / V geom: ", 1e5 * (Q1 ** 2).mean() / prod(lattice))
print("10^5 Q^2 / V angle: ", 1e5 * (Q2 ** 2).mean() / prod(lattice))
print("10^5 Q^2 / V gauge: ", 1e5 * (Q3 ** 2).mean() / prod(lattice))

# torch.save(torch.stack([Q1, Q2, Q3]), "top_charge.pt")

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True)
t = torch.arange(n)

ax1.plot(t, Q1, label="geometric")
ax2.plot(t, Q2, label="arg(z) plaquette")
ax3.plot(t, Q3, label="gauge plaquette")
ax3.set_xlabel("MD time")
ax1.legend()
ax2.legend()
ax3.legend()
ax1.set_title("Topological charge")

fig.savefig("top.png")

tau_top = IntegratedAutocorrelation(Q2)
tau = tau_top.compute().item()
t = range(0, min(n, round(10 * tau)))
fig, ax = plt.subplots()
ax.plot(t, tau_top.autocorrelation[t], label="autocorrelation")
ax.plot(t, tau_top.autocorrelation.cumsum(dim=-1)[t], label="cumulative")
ax.axvline(
    tau, ls="--", color="red", label=r"$\tau_{int}$"
)
ax.set_xlabel("MD time difference")
ax.set_title("Topological charge")
ax.legend()
fig.savefig("tau_top.png")
"""
