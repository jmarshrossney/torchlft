---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Free Scalar Fields in $d=2$ dimensions


$$
\begin{equation}
    S[\phi] = \frac{1}{2} \sum_{x,y\in\Lambda} \Sigma_{xy}^{-1} \phi_x \phi_y \, , \qquad
    \Sigma_{xy}^{-1} = (4 + m_0^2) \delta_{x,y} - \sum_{\mu=\pm 1}^{\pm 2} \delta_{x+\hat\mu, y}
\end{equation}
$$

In vector notation...

$$
\begin{equation}
    S[\phi] = \frac{1}{2} \underline{\phi}^\top \Sigma^{-1} \underline{\phi} \, , \qquad
    \Sigma^{-1} = -\delta^2 + m_0^2
\end{equation}
$$

where $\delta^2$ is the Laplacian for a $d=2$ dimensional square lattice.

```python
import torch
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-paper")
```

## From one to two dimensions


### One dimensional field

$$
\delta^2_{1d} = \begin{aligned}
    \begin{pmatrix}
    -2 & 1 & 0 & \ldots &  & 1\\
    1 & -2 & 1 & &  \\
    0 & 1 & -2 & 1 &  \\
    \vdots & & \ddots & \ddots & \ddots \\
    \\
    1 & & & & 1 & -2
	\end{pmatrix}
\end{aligned}
$$

```python
from torchlft.utils.lattice import laplacian

laplacian(6, 1)
```

```python
l = 8
m_sq = 0.25

K = -laplacian(l, 1) + m_sq * torch.eye(l)
Σ = torch.linalg.inv(K)
L = torch.linalg.cholesky(Σ)
L_inv = torch.linalg.cholesky(K)

fig, axes = plt.subplots(1, 4, figsize=(12, 12))
axes[0].set_title("Precision")
axes[1].set_title("Covariance")
axes[2].set_title("Cholesky/Cov")
axes[3].set_title("Cholesky/Prec")

axes[0].imshow(K)
axes[1].imshow(Σ)
axes[2].imshow(L)
axes[3].imshow(L_inv)

print(L)
print(L_inv)
```

```python
plt.plot(
    Σ[0]
)  # TODO mean for each shift, plot, show correlation length is 1/m
```

### Visualising the Kronecker product

```python
l = 4
M = torch.arange(l**2).view(l, l) + 1
L = M.tril()
δ_sq = -laplacian(l, 1)
Id = torch.eye(l)

fig, axes = plt.subplots(3, 3, figsize=(12, 9))

axes[0, 0].imshow(M)
axes[0, 1].imshow(torch.kron(Id, M))
axes[0, 2].imshow(torch.kron(M, Id))
axes[1, 0].imshow(L)
axes[1, 1].imshow(torch.kron(Id, L))
axes[1, 2].imshow(torch.kron(L, Id))
axes[2, 0].imshow(δ_sq)
axes[2, 1].imshow(torch.kron(Id, δ_sq))
axes[2, 2].imshow(torch.kron(δ_sq, Id))
```

### Building a two dimensional action


Kronecker product

$$
\begin{aligned}
    \delta^2_{2d} 
    = \delta^2_{1d} \otimes \mathbf{1} + \mathbf{1} \otimes \delta^2_{1d}
    = \begin{pmatrix}
    \delta^2_{1d} - 2 \mathbf{1} & -\mathbf{1} & \mathbf{0} & \ldots &  & -\mathbf{1}\\
    -\mathbf{1} & \delta^2_{1d} - 2 \mathbf{1} & -\mathbf{1} & &  \\
    \mathbf{0} & -\mathbf{1} & \delta^2_{1d} - 2 \mathbf{1} & -\mathbf{1} &  \\
    \vdots & & \ddots & \ddots & \ddots \\
    \\
    -\mathbf{1} & & & & -\mathbf{1} & \delta^2_{1d} - 2 \mathbf{1}
	\end{pmatrix}
\end{aligned}
$$

```python
print(torch.kron(Id, δ_sq).int())
print(torch.kron(δ_sq, Id).int())
# laplacian(4, 2).int()
```

```python
from torchlft.utils.lattice import restore_geometry_2d

L = 4
D = L * L
m_sq = 0.5

precision_matrix = -laplacian(L, 2) + m_sq * torch.eye(D)
covariance_matrix = torch.linalg.inv(precision_matrix)

fig, axes = plt.subplots(2, 2, figsize=(6, 6))
fig.suptitle(f"Scalar field on an {L}x{L} lattice")
axes[0, 0].set_ylabel("Precision")
axes[1, 0].set_ylabel("Covariance")
axes[1, 0].set_xlabel("Lexicographic")
axes[1, 1].set_xlabel("With 2d Geometry")

axes[0, 0].imshow(precision_matrix)
axes[0, 1].imshow(restore_geometry_2d(precision_matrix, (L, L)))
axes[1, 0].imshow(covariance_matrix)
axes[1, 1].imshow(restore_geometry_2d(covariance_matrix, (L, L)))
```

```python
from math import sqrt
from torchlft.utils.torch import log_cosh

L = 36
m_sq = 0.25

K = -laplacian(L, 2) + m_sq * torch.eye(L * L)
Σ = torch.linalg.inv(K)
C = restore_geometry_2d(Σ, (L, L))

log_G = C.sum(0).log()
ξ = 1 / sqrt(m_sq)
x = torch.linspace(0, L, 100)
log_cosh_ = log_cosh((x - L / 2) / ξ)
c = log_G[0] - log_cosh_[0]

fig, ax = plt.subplots()
ax.set_xlabel("x")
ax.set_ylabel("log sum_y Cov(x, y)")
ax.plot(x, log_cosh_ + c, "--", label="log cosh(m (x - L/2))")
ax.plot(log_G, "o", label="summed covariance")
ax.legend()
```

## Fourier Transform


The discrete Fourier Transform (DFT) 

$$
\begin{aligned}
    \mathcal{F}(\phi)_k = \sum_{x\in\Lambda} e^{i k \cdot x} \phi_x \, , \qquad
    k \cdot x \equiv \sum_{\mu=1}^2 k_\mu x_\mu
\end{aligned}
$$

diagonalises the Laplacian matrix,

$$
\begin{aligned}
    S[\phi] = \frac{1}{2} \sum_{x,y\in\Lambda} \Sigma_{xy}^{-1} \phi_x \phi_y
    = \frac{1}{2} \sum_{k\in \widetilde{\Lambda}} \lambda_k | \mathcal{F}(\phi)_k |^2 \, , \qquad
\end{aligned}
$$

where $\lambda_k$ are eigenvalues labelled by the corresponding momentum vector $k = (k_1, k_2)$, and defined by 

$$
\begin{aligned}
    \lambda_k = \frac{1}{|\Lambda|} \left[ m_0^2 + \sum_{\mu=1}^2 \hat{k}_\mu \right] \, , \qquad \hat{k}_\mu = 4 \sin^2 \frac{k_\mu}{2}
\end{aligned}
$$

Assuming an $L \times L$ square lattice with $a=1$ there are $L^2 / 2 + L/2 = \frac{L(L+1)}{2}$ unique eigenvalues.

```python
# TODO
# My old code which generated free fields via inverse FT

l = 4
D = l * l
m_sq = 0.1

k = torch.stack(
    torch.meshgrid(torch.arange(l), torch.arange(l)),
    dim=-1,
)
k_hat = (k / 2).sin().pow(2) * 4

λ = (1 / D) * (m_sq + k_hat.sum(dim=-1))

λ

# K = -laplacian(l, 2) + m_sq * torch.eye(D)
```

```python
import torch
from torch.fft import fft, rfft, rfft2

L = 8
x = torch.randn(L, L)

y1 = rfft2(x)
y2 = fft(rfft(x, dim=1), dim=0)

print(y1 - y2)
# print(y2)
```

The DFT transforms real-valued fields into complex Hermitean fields in Fourier space.

But we want to deal with real vectors $\underline{\phi} \in \mathbb{R}^{D}$ with $D = \lvert\Lambda\rvert$.

- Also, DFT is equivalent to discrete convolution.
- Translation invariance: block circulant
- DFT is unitary, so assert that $F$ is orthogonal

$$
\begin{aligned}
    \underline{\phi}^\top (-\delta^2 + m_0^2) \underline{\phi}
    &= \underline{\phi}^\top F^\top F (-\delta^2 + m_0^2) F^\top F \underline{\phi} \\
    &= (F \underline{\phi})^\top (-F \delta^2 F^{\top} + m_0^2) (F \underline{\phi}) \\
    &= (F \underline{\phi})^\top \mathrm{diag}(\lambda_1, \ldots, \lambda_D) (F \underline{\phi})
\end{aligned}
$$

Note that $\mathcal{F}(\phi)$ still has a 2d geometry; it is an object indexed by a tuple on the reciprocal lattice $k = (k_1, k_2) \in \widetilde{\Lambda}$ which label the momenta.

On the other hand, $F \phi$ is a one-dimensional vector obtained by acting on a lexicographically-ordered vector $\phi$ with a linear transformation $F$.
It doesn't have any geometry to speak of; just as all permutations of the eigenvalues $\mathrm{diag}(\lambda_1, \ldots, \lambda_D)$ are equivalent, so is any ordering of the elements of $F\phi$.


## Cholesky Decomposition


The Cholesky decomposition of a square positive definite symmetric matrix $\Sigma$ is $\Sigma = L^\top L$ where $L$ is a (lower) triangular matrix.

This implies that the action can be rewritten

$$
\begin{aligned}
    \underline{\phi}^\top \Sigma^{-1} \underline{\phi}
    &= \underline{\phi}^\top (L^\top L)^{-1} \underline{\phi} \\
    &= (L^{-1} \underline{\phi})^\top (L^{-1} \underline{\phi})
\end{aligned}
$$

If we draw uncorrelated Gaussian numbers $\underline{z} \sim \mathcal{N}(0, \mathbf{1}_D)$ and transform them

$$
\begin{aligned}
    \underline{\phi} = L \underline{z}
\end{aligned}
$$

Then the result will be a Gaussian vector with null mean and covariance equal to $\Sigma$.

We can see this using the change of variables.

$$
\begin{aligned}
    p(\underline{\phi}) &= q\left(L^{-1}\underline{\phi} \right) \left\lvert \frac{\partial L^{-1}\underline{\phi}}{\partial \underline{\phi}} \right\rvert \\
    &=  \frac{1}{\sqrt{(2\pi)^n}} \exp\left( -\frac{1}{2} (L^{-1}\underline{\phi})^\top L^{-1} \underline{\phi} \right) \cdot \left\lvert L^{-1} \right\rvert \\
    &= \frac{\sqrt{|\Sigma^{-1}|}}{\sqrt{(2\pi)^n}} \exp\left( -\frac{1}{2} \underline{\phi}^\top \Sigma^{-1} \underline{\phi} \right) \\
    &= \frac{1}{\sqrt{| 2\pi \Sigma |}} \exp\left( -\frac{1}{2} \underline{\phi}^\top \Sigma^{-1} \underline{\phi} \right)
\end{aligned}
$$

We used that  $|\Sigma^{-1}| = |L^{-1}|^2$ and $|c A| = c^D |A|$ for an $D \times D$ matrix.

```python
l = 4
D = l * l
m_sq = 0.5

K = -laplacian(l, 2) + m_sq * torch.eye(D)
Σ = torch.linalg.inv(K)
L = torch.linalg.cholesky(Σ)
L_inv = torch.linalg.cholesky(K)

# print(L)

print("Doesn't seem to round trip..? Inv[Chol[K]] =/= Chol[Inv[K]]")
print((L_inv - torch.linalg.inv(L)).abs().max())

# NOTE: L L^\top here rather than L^\top L
assert torch.allclose(L @ L.T, Σ)

fig, axes = plt.subplots(2, 2, figsize=(6, 6))
fig.suptitle(f"Cholesky decomposition")
axes[0, 0].set_ylabel("L")
axes[1, 0].set_ylabel("L inverse")
axes[1, 0].set_xlabel("Matrix")
axes[1, 1].set_xlabel("Squared")

axes[0, 0].imshow(L)
axes[0, 1].imshow(L @ L.T)
axes[1, 0].imshow(L_inv)
axes[1, 1].imshow(L_inv @ L_inv.T)
```

How does the Cholesky decompistion relate to the DFT matrix from earlier?

Equating the two,

$$
\begin{aligned}
    (F \underline{\phi})^\top \mathrm{diag}(\lambda_1, \ldots, \lambda_D) (F \underline{\phi})
    = \underline{\phi}^\top \Sigma^{-1} \underline{\phi}
    = (L^{-1} \underline{\phi})^\top (L^{-1} \underline{\phi})
\end{aligned}
$$

Hence

$$
\begin{aligned}
    L^{-1} = F \, \mathrm{diag}(\sqrt{\lambda_1}, \ldots, \sqrt{\lambda_D})
\end{aligned}
$$

```python
l = 4
D = l * l
m_sq = 0.1


K = -laplacian(l, 2) + m_sq * torch.eye(D)
Σ = torch.linalg.inv(K)
L = torch.linalg.cholesky(Σ)
L_inv = torch.linalg.cholesky(K)

λ, v = torch.linalg.eig(L_inv)
print(λ)
```

Is this equivariant under translations?

No!

```python
l = 4
D = l * l
m_sq = 0.1

K = -laplacian(l, 2) + m_sq * torch.eye(D)
Σ = torch.linalg.inv(K)
L = torch.linalg.cholesky(Σ)
L_inv = torch.linalg.cholesky(K)

z = torch.empty(l, l).normal_()

φ_1 = torch.mv(L, z.flatten()).view(l, l)

φ_2 = torch.mv(L, z.roll(1, 0).flatten()).view(l, l).roll(-1, 0)

print(φ_1)
print(φ_2)

φ_1 = φ_1.flatten()
φ_2 = φ_2.flatten()
print(torch.dot(φ_1, torch.mv(K, φ_1)))
print(torch.dot(φ_2, torch.mv(K, φ_2)))
```

### Spectrum of Choleksy versus Covariance

Note that eigvals of Cholesky are not the square root of eigvals of covariance!
You would need to FT to make the cov diagonal for this to be true.

```python
l = 8
D = l * l
m_sq = 0.25

K = -laplacian(l, 2) + m_sq * torch.eye(D)
Σ = torch.linalg.inv(K)
L = torch.linalg.cholesky(Σ)
L_inv = torch.linalg.cholesky(K)


λ_chol = torch.linalg.eigvals(L).real
λ_chol, _ = λ_chol.sort()

cov = Σ
λ_cov = torch.linalg.eigvals(cov).real
λ_cov, _ = λ_cov.sort()

D = len(λ_cov)

fig, ax = plt.subplots()
ax.plot(range(D), λ_chol, "o--", label="chol")
ax.plot(range(D), λ_cov.sqrt(), "o--", label="cov")

ax.legend()
```

## Learning the Cholesky Decomposition

```python
import torch
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-paper")

from torchlft.scripts.train import parser, main
```

```python
config = """
model:
  class_path: torchlft.model_zoo.gaussian.TriangularLinearModel
  init_args: 
    target:
      lattice_length: {L}
      lattice_dim: {d}
      m_sq: {m_sq}

train:
    n_steps: 2000
    batch_size: 2000
    init_lr: 0.01
    display_metrics: false

output: false
"""
config = parser.parse_string(config.format(L=8, d=2, m_sq=0.25))
print(parser.dump(config, skip_none=False))
```

```python
model, logger = main(config)
```

```python
from plot import plot_metrics

fig = plot_metrics(logger)
```

```python
expected_weights = model.target.cholesky
empirical_weights = model.transform.get_weight().detach()

fig, axes = plt.subplots(1, 3, figsize=(9, 3))

axes[0].set_axis_off()

axes[0].imshow(expected_weights)
axes[1].imshow(empirical_weights)
axes[2].imshow(empirical_weights - expected_weights)
```

```python
expected_weights = model.target.cholesky
empirical_weights = model.transform.get_weight().detach()

fig, axes = plt.subplots(1, 3, figsize=(8, 3))

axes[0].set_axis_off()
axes[0].set_title(r"$L_\theta$")
axes[1].set_axis_off()
axes[1].set_title(r"$L_\theta - L_\ast$")
axes[2].set_yticks([])
axes[2].set_yticklabels([])
axes[2].set_xlim(-0.0035, 0.0035)
axes[2].set_xlabel(r"$L_\theta - L_\ast$")

axes[0].imshow(empirical_weights)
axes[1].imshow(empirical_weights - expected_weights)
axes[2].hist(
    (empirical_weights - expected_weights)[model.transform.mask], bins=35
)

fig.tight_layout()
fig.savefig("cholesky_heatmaps.png", dpi=300)
```

```python
_ = plt.hist(
    (empirical_weights - expected_weights)[model.transform.mask].flatten(),
    bins=25,
)
```

```python
from torchlft.nflow.utils import get_model_jacobian

jac, inputs, outputs = get_model_jacobian(model, 1)

assert torch.allclose(jac.squeeze(0), empirical_weights)
```

```python
expected_cov = model.target.covariance

sample, weights = model.weighted_sample(10000)

empirical_cov = torch.cov(sample.transpose(0, 1))

_ = plt.hist((expected_cov - empirical_cov).flatten(), bins=25)
```

```python
sample, indices = model.metropolis_sample(10000)
indices = indices.tolist()
print("Acceptance: ", len(set(indices)) / len(indices))
```
