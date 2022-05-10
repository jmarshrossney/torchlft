from __future__ import annotations

import torch


def phi_four_action(
    sample: torch.Tensor, m_sq: torch.Tensor, lam: torch.Tensor
) -> torch.Tensor:
    r"""Phi^4 action for a single-component scalar field.

    The action is defined as follows

    .. math::

        S(\phi) = \sum_{x\in\Lambda} \left[
            -\sum_{\mu=1}^d \phi(x) \phi(x+e_\mu)
            + \frac{4 + m^2}{2} \phi(x)^2
            + \lambda \phi(x)^4
        \right]


    Parameters
    ----------
    sample
        Sample of field configurations with shape
        ``(batch_size, 1, *lattice_shape)``
    m_sq
        Bare mass, squared
    lam
        Quartic coupling strength


    See Also
    --------
    TODO: convert between parametrisations

    """
    assert sample.shape[1] == 1, "Expected a single component scalar field"
    phi = sample
    action_density = torch.zeros_like(phi)

    # Nearest neighbour interaction
    for dim in range(2, phi.dim()):
        action_density.sub_(phi.mul(phi.roll(-1, dim)))

    # phi^2 term
    phi_sq = phi.pow(2)
    action_density.add_(phi_sq.mul((4 + m_sq) / 2))

    # phi^4 term
    action_density.add_(phi_sq.pow(2).mul(lam))

    # Sum over lattice sites
    return action_density.flatten(start_dim=1).sum(dim=1)


def xy_nn_action(sample: torch.Tensor, coupling: torch.Tensor) -> torch.Tensor:
    r"""Action for the classical XY model with pure nearest-neighbour coupling.

    .. math::

        S(\phi) = - J \sum_{x\in\Lambda} \sum_{\mu=1}^d
            \cos(\phi_{x+e_\mu} - \phi_x)

    Parameters
    ----------
    sample
        Sample of field configurations with shape
        ``(batch_size, 1, *lattice_shape)``
    coupling
        Nearest-neighbour coupling constant :math:`J`
    """
    assert sample.shape[1] == 1, "Expected a single component field"
    phi = sample
    action_density = torch.zeros_like(phi)

    for dim in range(2, phi.dim()):
        action_density.sub_(phi.roll(1, dim).sub(phi).cos().mul(coupling))

    return action_density.flatten(start_dim=1).sum(dim=1)
