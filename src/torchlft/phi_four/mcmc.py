import torch


def hmc(phi: torch.Tensor, action, *, tau: float, n_steps: int) -> bool:
    phi_cp = torch.clone(phi).detach()

    phi.requires_grad = True
    phi.grad = torch.zeros(phi.shape)  # initialize gradient

    # Initialize momenta
    mom = torch.randn(phi.shape)

    # Initial Hamiltonian
    H_0 = hamiltonian(mom, phi, action)

    # Leapfrog integrator
    leapfrog(mom, phi, action, tau=tau, n_steps=n_steps)

    # Final Hamiltonian
    dH = hamiltonian(mom, phi, action) - H_0

    if dH > 0:
        if torch.rand(1).item() >= torch.exp(-torch.Tensor([dH])).item():
            with torch.no_grad():
                phi[:] = phi_cp  # element-wise assignment
            return True

    return False


def hamiltonian(mom, phi, action):
    """
    Computes the Hamiltonian of `hmc` function.
    """
    H = 0.5 * torch.sum(mom ** 2) + action(phi)

    return H.item()


def leapfrog(mom, phi, action, *, tau, n_steps):
    dt = tau / n_steps

    load_action_gradient(phi, action)
    mom -= 0.5 * dt * phi.grad

    for i in range(n_steps):
        with torch.no_grad():
            phi += dt * mom

        if i == n_steps - 1:
            load_action_gradient(phi, action)
            mom -= 0.5 * dt * phi.grad
        else:
            load_action_gradient(phi, action)
            mom -= dt * phi.grad


def load_action_gradient(phi, action):
    """
    Passes `phi` through fucntion `action`and loads the gradient with respect
    to the initial fields into `phi.grad`, without overwriting `phi`.
    """
    phi.grad.zero_()

    S = action(phi)

    external_grad_S = torch.ones(S.shape)

    S.backward(gradient=external_grad_S)
