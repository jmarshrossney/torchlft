import dataclasses
from typing import Union

import torch


# TODO torchscript
def _phi_four_action(
    sample: torch.Tensor,
    hopping: Union[float, torch.Tensor],
    quad: Union[float, torch.Tensor],
    quart: Union[float, torch.Tensor],
) -> torch.Tensor:
    """Computes the Phi^4 action for a single-component scalar field."""
    phi = sample
    action_density = torch.zeros_like(phi)

    # Nearest neighbour interaction
    for dim in range(1, phi.dim()):
        action_density.sub_(phi.mul(phi.roll(-1, dim)).mul(hopping))

    # phi^2 term
    phi_sq = phi.pow(2)
    action_density.add_(phi_sq.mul(quad))

    # phi^4 term
    action_density.add_(phi_sq.pow(2).mul(quart))

    # Sum over lattice sites
    return action_density.flatten(start_dim=1).sum(dim=1)


def phi_four_action_standard(
    sample: torch.Tensor,
    m_sq: Union[float, torch.Tensor],
    lam: Union[float, torch.Tensor],
) -> torch.Tensor:
    r"""Phi^4 action for a single-component scalar field.

    The standard Particle Physics parametrisation of the phi^4
    action on the lattice is

    .. math::

        S(\phi) = \sum_{x\in\Lambda} \left[
            -\sum_{\mu=1}^d \phi(x) \phi(x+e_\mu)
            + \frac{4 + m^2}{2} \phi(x)^2
            + \lambda \phi(x)^4
        \right]

    Args:
        sample
            Sample of field configurations with shape
            ``(batch_size, *lattice_shape)``
        m_sq
            Bare mass, squared, :math:`m^2`
        lam
            Quartic coupling strength, :math:`lambda`

    .. seealso:: :func:`phi_four_action_ising`
    """
    return _phi_four_action(sample, hopping=1, quad=(4 + m_sq) / 2, quart=lam)


def phi_four_action_ising(
    sample: torch.Tensor,
    beta: Union[float, torch.Tensor],
    lam: Union[float, torch.Tensor],
) -> torch.Tensor:
    r"""Phi^4 action with the alternative Ising-like parametrisation.

    A phi^4 action that explicitly reduces to the Ising model in the
    limit of infinite quartic self-coupling is given by

    .. math::

        S(\phi) = \sum_{x\in\Lambda} \left[
            -\beta \sum_{\mu=1}^d \phi(x) \phi(x+e_\mu)
            + \phi(x)^2
            + \lambda (\phi(x)^2 - 1)^2
        \right]

    This function actually computes the following

    .. math::

        S(\phi) = \sum_{x\in\Lambda} \left[
            -\beta \sum_{\mu=1}^d \phi(x) \phi(x+e_\mu)
            + (1 - 2\lambda) \phi(x)^2
            + \lambda \phi(x)^4
        \right]

    which is equivalent up to a constant shift that does not
    affect the physics.

    Args:
        sample
            Sample of field configurations with shape
            ``(batch_size, *lattice_shape)``
        beta
            Inverse temperature, :math:`beta`
        lam
            Quartic coupling strength, :math:`lambda`

    .. note::

        Often in the literature one sees :math:`2\kappa` in place of
        :math:`beta`, with :math:`\kappa` referred to as the 'hopping'
        parameter.

    .. seealso:: :func:`phi_four_action_standard`
    """
    return _phi_four_action(sample, hopping=beta, quad=1 - 2 * lam, quart=lam)


@dataclasses.dataclass
class PhiFourActionStandard:
    """
    Class-based implementation of :func:`phi_four_action_standard`.

    This class can serve as a 'target distribution' in e.g. a Normalizing Flow,
    through the ``log_prob`` method. See :py:class:`torchnf.abc.TargetDistribution`.
    """

    m_sq: Union[float, torch.Tensor]
    lam: Union[float, torch.Tensor]

    def action(self, sample: torch.Tensor) -> torch.Tensor:
        """Calls ``phi_four_action_standard`` with the sample provided."""
        return phi_four_action_standard(sample, self.m_sq, self.lam)

    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        """Returns the negated action."""
        return self.action(sample).neg()


@dataclasses.dataclass
class PhiFourActionIsing:
    """
    Class-based implementation of :func:`phi_four_action_ising`.

    This class can serve as a 'target distribution' in e.g. a Normalizing Flow,
    through the ``log_prob`` method. See :py:class:`torchnf.abc.TargetDistribution`.
    """

    beta: Union[float, torch.Tensor]
    lam: Union[float, torch.Tensor]

    def action(self, sample: torch.Tensor) -> torch.Tensor:
        """Calls ``phi_four_action_ising`` with the sample provided."""
        return phi_four_action_ising(sample, self.beta, self.lam)

    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        """Returns the negated action."""
        return self.action(sample).neg()
