import logging
from math import isclose

import torch
import torch.linalg as LA

from torchlft.typing import Callable, Optional, Tensor, Hamiltonian

log = logging.getLogger(__name__)

# TODO: typing for Hamiltonian


# TODO: adaptive step size (on_step_func callback)
def leapfrog(
    x0: Tensor,
    p0: Tensor,
    hamiltonian: Hamiltonian,
    *,
    step_size: float,
    traj_length: float,
    on_step_func: Optional[Callable[[Tensor, Tensor, float], None]] = None,
) -> tuple[Tensor, Tensor, float]:
    n_steps = max(1, round(traj_length / abs(step_size)))
    if not isclose(n_steps * step_size, traj_length):
        log.warning("to do")

    x = x0.clone()
    p = p0.clone()
    t = 0
    ε = step_size

    F = hamiltonian.grad_potential(x).negative()

    for _ in range(n_steps):
        if on_step_func is not None:
            on_step_func(x, p, t)

        # NOTE: avoid in-place here in case p stored in on_step_func
        p = p + (ε / 2) * F

        v = hamiltonian.grad_kinetic(p)

        x = x + ε * v

        F = hamiltonian.grad_potential(x).negative()

        p += (ε / 2) * F

        t += ε

    return x, p, t


# NOTE: could make this a class to combine with leapfrog in R^N
# just substitute the drift update drift(x, p, v) -> (x', p')


def leapfrog_sphere(
    x0: Tensor,
    p0: Tensor,
    hamiltonian: Hamiltonian,
    *,
    step_size: float,
    traj_length: float,
    on_step_func: Optional[Callable[[Tensor, Tensor, float], None]] = None,
) -> tuple[Tensor, Tensor, float]:
    n_steps = max(1, round(traj_length / abs(step_size)))
    if not isclose(n_steps * step_size, traj_length):
        log.warning("to do")

    x = x0.clone()
    p = p0.clone()
    t = 0
    ε = step_size

    F = hamiltonian.grad_potential(x).negative()

    for _ in range(n_steps):
        if on_step_func is not None:
            on_step_func(x, p, t)

        # NOTE: avoid in-place here in case p stored in on_step_func
        p = p + (ε / 2) * F

        v = hamiltonian.grad_kinetic(p)

        mod_p = LA.vector_norm(p, dim=-1, keepdim=True)
        mod_v = LA.vector_norm(v, dim=-1, keepdim=True)
        cos_εv = (ε * mod_v).cos()
        sin_εv = (ε * mod_v).sin()

        p = cos_εv * p - (sin_εv * mod_p) * x
        x = cos_εv * x + (sin_εv / mod_v) * v

        mod_x = LA.vector_norm(x, dim=-1, keepdim=True)
        x /= mod_x

        F = hamiltonian.grad_potential(x).negative()

        p += (ε / 2) * F

        t += ε

    return x, p, t


# TODO: Euler or runge kutta for gradient flow
