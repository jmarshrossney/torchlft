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

# Coupling Layer Models and Gaussian Fields

```python
import torch
import matplotlib.pyplot as plt

from torchlft.scripts.train import parser, main
from torchlft.nflow.utils import get_jacobian, get_model_jacobian
from torchlft.nflow.layer import Layer

plt.style.use("seaborn-v0_8-paper")
```

## Define some plotting functions

```python
def plot_jacobian_squared_vs_covariance(model_, batch_size=64):
    with torch.no_grad():
        jac, _, _ = get_model_jacobian(model_, batch_size)

    print(jac)
    jac_sq = torch.einsum("bij,bkj-> bik", jac, jac)
    cov = model_.target.covariance

    fig, axes = plt.subplots(1, 3)

    axes[0].imshow(cov)
    axes[1].imshow(jac_sq.mean(dim=0))
    im = axes[2].imshow(jac_sq.mean(dim=0) - cov)
    fig.colorbar(im)

    return fig


def plot_qr_diag(model_):
    cholesky = model_.target.cholesky
    λ_chol = cholesky.diag()
    λ_chol, _ = λ_chol.sort()

    with torch.no_grad():
        jac, _, _ = get_model_jacobian(model_, 1)

    Q, R = torch.linalg.qr(jac[0])
    λ_R = R.diag().abs()
    λ_R, _ = λ_R.sort()

    fig, ax = plt.subplots()

    ax.plot(λ_chol, "k--", label="Cholesky")
    ax.plot(λ_R, "ro:", label="Model")
    ax.legend()

    return fig


def plot_qr_vs_cholesky(model_):
    cholesky = model_.target.cholesky

    with torch.no_grad():
        jac, _, _ = get_model_jacobian(model_, 1)

    Q, R = torch.linalg.qr(jac[0])

    print(cholesky + R.T)
    # print(R)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(R.T)
    ax2.imshow(cholesky)
    im = ax3.imshow(R.T - cholesky)
    fig.colorbar(im)

    return fig


def plot_jacobian_spectrum(model_, batch_size=64):
    cholesky = model_.target.cholesky
    λ_chol = torch.linalg.eigvals(cholesky).real
    λ_chol, _ = λ_chol.sort()

    with torch.no_grad():
        jac, _, _ = get_model_jacobian(model_, batch_size)

    Q, R = torch.linalg.qr(jac)
    λ = torch.linalg.eigvals(R)
    _, D = λ.shape

    λ_real, λ_imag, λ_abs = λ.real, λ.imag, λ.real.abs()

    print(λ_imag.abs().max())

    λ_real, _ = λ_real.sort(dim=1)
    λ_imag, _ = λ_imag.sort(dim=1)
    λ_abs, _ = λ_abs.sort(dim=1)

    fig, axes = plt.subplots(1, 2, figsize=(8, 6))
    axes = iter(axes.flatten())

    ax = next(axes)
    # ax.fill_between(range(D), λ_real.quantile(0., dim=0), λ_real.quantile(1., dim=0), alpha=0.7, label="Real")
    # ax.plot(range(D), λ_real.quantile(0.5, dim=0), "r-", label="Real")
    # print(λ_real)
    # print(λ_imag)

    # ax.fill_between(range(D), λ_imag.quantile(0., dim=0), λ_imag.quantile(1., dim=0), alpha=0.7, label="Imag")
    # ax.plot(range(D), λ_imag.quantile(0.5, dim=0), "g-", label="Imag")
    ax.plot(range(D), λ_abs.quantile(0.5, dim=0), "b-", label="Abs(real)")

    ax.plot(range(D), λ_chol, "k--", label="Cholesky")

    ax.legend()

    ax = next(axes)
    det_expec = torch.linalg.det(cholesky)
    det = torch.linalg.det(jac)
    # ax.hist(det.log(), label="model")
    ax.axvline(det_expec.log(), label="Choleksy", color="r")

    eig_prod = torch.prod(λ, dim=1)
    ax.hist(eig_prod.real.abs().log(), label="Real product of eigs")
    max_imag_log_det = eig_prod.imag.clamp(0).log().max()
    print(f"Max imaginary part: {max_imag_log_det}")

    ax.legend()

    return fig


def plot_layer_jacobian_determinants(model_, batch_size=64):
    layers = [
        (name, mod)
        for name, mod in model_.named_modules()
        if isinstance(mod, Layer)
    ]
    # print(layers)

    flow, *layers = layers

    inputs, _ = model_.sample_base(batch_size)
    cloned_inputs = inputs.clone()

    dets, ldjs = [], []

    for label, transform in layers:

        with torch.no_grad():
            jac, _, _ = get_jacobian(transform, inputs)
            outputs, ldj = transform(inputs)

            inputs = outputs

        dets.append(torch.linalg.det(jac))
        ldjs.append(ldj.squeeze(-1))

    dets = torch.stack(dets)
    dets = dets.log()

    ldjs = torch.stack(ldjs)
    print((dets - ldjs).abs().max())

    fig, ax = plt.subplots()

    with torch.no_grad():
        jac, _, _ = get_jacobian(flow[1], cloned_inputs)

    # ldj = torch.linalg.det(jac).log()
    # ax.axhspan(ldj.quantile(0.25), ldj.quantile(0.75), color="red", alpha=0.5, label="smee")
    # ax.axhline(ldj.quantile(0.5), color="red", label="full flow")
    correct = model_.target.cholesky.diag().log().sum()
    ax.axhline(correct, color="red", label="log det Chol")

    dets = dets.cumsum(dim=0)
    ax.fill_between(
        range(len(layers)),
        dets.quantile(0.25, dim=1),
        dets.quantile(0.75, dim=1),
        alpha=0.5,
    )
    ax.plot(
        range(len(layers)), dets.quantile(0.5, dim=1), "o--", label="layers"
    )

    # [flow, ("full model", model_.flow_forward)]

    ax.legend()
    fig.tight_layout()
    return fig


def plot_metrics(logger):
    metrics = logger.get_data()

    steps = metrics["steps"]
    kl_div = -metrics["mlw"]
    one_minus_ess = 1 - metrics["ess"]
    one_minus_acc = 1 - metrics["acc"]
    vlw = metrics["vlw"]

    print("Mean acceptance: ", float(metrics["acc"].mean(1)[-1]))

    def plot(ax, tensor):
        q = torch.tensor([0.0, 1.0], dtype=tensor.dtype)
        ax.fill_between(steps, *tensor.quantile(q, dim=1), alpha=0.5)
        ax.plot(steps, tensor.quantile(0.5, dim=1))
        ax.set_yscale("log")

    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(8, 6))

    axes = iter(axes.flatten())

    ax = next(axes)
    plot(ax, kl_div)
    ax.set_title("KL Divergence")

    ax = next(axes)
    plot(ax, one_minus_acc)
    ax.set_title("1 - Acceptance")

    ax = next(axes)
    plot(ax, one_minus_ess)
    ax.set_title("1 - ESS")

    ax = next(axes)
    plot(ax, vlw)
    ax.set_title("Var log weights")

    fig.tight_layout()

    return fig
```

## Configuration for Dense Coupling Model

```python
config_str = """
model:
  class_path: torchlft.model_zoo.gaussian.DenseCouplingModel
  init_args:
    target:
      lattice_length: {L}
      lattice_dim: 2
      m_sq: {m_sq}
    
    flow:
      transform:
        scale_fn: exponential
        symmetric: {symmetric}
        shift_only: {shift_only}
      net:
        sizes: {sizes}
        activation: {activation}
        bias: {bias}
      n_blocks: {depth}
      final_layer_bias: {bias}
      final_diagonal_layer: {final_diagonal}
      
    partitioning: {partitioning}

train:
  n_steps: {steps}
  batch_size: {batch}
  init_lr: 0.005
  display_metrics: false

output: false
cuda: true
"""


def get_config(
    L: int = 8,
    m_sq: float = 0.25,
    sizes: list[int] = [64],
    depth: int = 1,
    symmetric: bool = False,
    shift_only: bool = False,
    partitioning: str = "checkerboard",
    final_diagonal: bool = False,
    bias: bool = False,
    activation: str = "tanh",
    steps: int = 2000,
    batch: int = 1024,
):
    config = config_str.format(
        L=L,
        m_sq=m_sq,
        sizes=sizes,
        depth=depth,
        symmetric=symmetric,
        shift_only=shift_only,
        partitioning=partitioning,
        final_diagonal=final_diagonal,
        bias=bias,
        activation=activation,
        steps=steps,
        batch=batch,
    )
    return parser.parse_string(config)
```

```python
# TEST
config = get_config()
print(parser.dump(config))

instantiated_config = parser.instantiate_classes(config)
untrained_model = instantiated_config.model
_ = untrained_model(1)
print(untrained_model)
```

## Linear Model

```python
config = get_config(
    depth=2,
    sizes=[],
    partitioning="lexicographic",
    shift_only=True,
    final_diagonal=True,
    bias=False,
    activation="identity",
)

instantiated_config = parser.instantiate_classes(config)
untrained_model = instantiated_config.model
_ = untrained_model(1)
print(untrained_model)
```

<!-- #raw -->
_ = plot_layer_jacobians(untrained_model)
<!-- #endraw -->

```python
trained_model, logger = main(config)

acceptance = float(logger.get_data()["acc"].mean(dim=1)[-1])
print(f"Final acceptance: {acceptance}")

_ = plot_qr_vs_cholesky(trained_model)
# _ = plot_qr_diag(trained_model)
# _ = plot_layer_jacobians(trained_model)
_ = plot_jacobian_squared_vs_covariance(trained_model, 1)
# _ = plot_jacobian_spectrum(trained_model)
# _ = plot_layer_jacobian_determinants(trained_model)
```

```python
def plot_qr_diag_many_models(models_):

    cholesky = list(models_.values())[0].target.cholesky
    λ_chol = cholesky.diag()
    λ_chol, _ = λ_chol.sort()

    fig, ax = plt.subplots()

    (c,) = ax.plot(λ_chol, "k-", label="Cholesky")

    handles, labels = [c], ["Cholesky eigvals"]

    for label, model_ in models_.items():

        with torch.no_grad():
            jac, _, _ = get_model_jacobian(model_, 1)

        Q, R = torch.linalg.qr(jac[0])
        λ_R = R.diag().abs()
        λ_R, _ = λ_R.sort()

        # Marker to indicate which one is negative??

        (line,) = ax.plot(λ_R)

        labels.append(label)
        handles.append(line)

    ax.set_xlabel("Ascending order")
    ax.set_ylabel("Absolute value of real eigenvalues")

    ax.legend(handles=handles, labels=labels)

    return fig
```

```python
models = {}

for steps in (100, 500, 1000, 5000, 10000):
    config = get_config(
        depth=2,
        sizes=[],
        partitioning="lexicographic",
        shift_only=True,
        final_diagonal=True,
        bias=False,
        activation="identity",
        steps=steps,
        batch=8192,
    )
    trained_model, logger = main(config)
    mlw = float(logger.get_data()["mlw"].mean(dim=1)[-1])

    label = f"{steps} steps, (KL: {-mlw:.2g})"
    models[label] = trained_model

fig = plot_qr_diag_many_models(models)
```

## Non-linear Model

```python
config = get_config(
    depth=6,
    sizes=[64],
    partitioning="lexicographic",
    shift_only=False,
    final_diagonal=False,
    bias=True,
    activation="tanh",
)

instantiated_config = parser.instantiate_classes(config)
untrained_model = instantiated_config.model
_ = untrained_model(1)
print(untrained_model)
```

```python
trained_model, logger = main(config)

acceptance = float(logger.get_data()["acc"].mean(dim=1)[-1])
print(f"Final acceptance: {acceptance}")

# _ = plot_layer_jacobians(trained_model)
_ = plot_jacobian_squared_vs_covariance(trained_model, 1)
_ = plot_jacobian_spectrum(trained_model)
_ = plot_layer_jacobian_determinants(trained_model)
```

## Compare models of different depth

```python
def plot_ldj_multiple_flows(models: dict, batch_size=64):

    assert (
        len(set([model.target.lattice_length for model in models.values()]))
        == 1
    )
    assert len(set([model.target.m_sq for model in models.values()])) == 1

    data = {}

    for label, model in models.items():
        layers = [
            (name, mod)
            for name, mod in model.named_modules()
            if isinstance(mod, Layer)
        ]
        _, *layers = layers

        inputs, _ = model.sample_base(batch_size)

        ldjs = []

        for _, transform in layers:
            with torch.no_grad():
                outputs, ldj = transform(inputs)
                inputs = outputs

            ldjs.append(ldj.squeeze(-1))

        ldjs = torch.stack(ldjs)

        data[label] = ldjs

    fig, ax = plt.subplots()

    correct = model.target.cholesky.diag().log().sum()
    hl = ax.axhline(correct, color="black", linestyle=":")

    legend = [[hl, "log det Cholesky"]]

    for label, ldj in data.items():
        depth = range(1, len(ldj) + 1)
        print(ldj.shape)
        ldj = ldj.cumsum(dim=0)

        fb = ax.fill_between(
            depth,
            ldj.quantile(0.25, dim=1),
            ldj.quantile(0.75, dim=1),
            alpha=0.5,
        )
        (l,) = ax.plot(depth, ldj.quantile(0.5, dim=1), "--")
        handle = (l, fb)
        legend.append([handle, label])

    handles, labels = list(zip(*legend))
    ax.legend(handles=handles, labels=labels)

    ax.set_ylabel("Cumulative log det Jacobian")
    ax.set_xlabel("Layer")

    fig.tight_layout()
    return fig
```

```python
configs = {}

# configs["1x Linear Coupling + Diagonal"] = get_config(depth=1, sizes=[], partitioning="lexicographic", shift_only=True, final_diagonal=True, bias=False, activation="identity")
configs["2x Linear Coupling + Diagonal"] = get_config(
    depth=2,
    sizes=[],
    partitioning="lexicographic",
    shift_only=True,
    final_diagonal=True,
    bias=False,
    activation="identity",
)

defaults = dict(
    sizes=[64],
    partitioning="lexicographic",
    shift_only=False,
    final_diagonal=False,
    bias=True,
    activation="tanh",
)

for depth in (2, 4, 6, 8):
    configs[f"{depth}x Non-linear Coupling"] = get_config(
        depth=depth, **defaults
    )


models = {}

for label, config in configs.items():

    trained_model, logger = main(config)

    ess = float(logger.get_data()["ess"].mean(dim=1)[-1])
    label += f" (ESS: {ess:.2f})"

    models[label] = trained_model

fig = plot_ldj_multiple_flows(models)
```

```python
fig.savefig("depth_convergence.png", dpi=300)
```

```python

```

## Configuration for Convolutional Model

```python
config = """
model:
  class_path: torchlft.model_zoo.gaussian.ConvCouplingModel
  init_args:
    target:
      lattice_length: 12
      lattice_dim: 2
      m_sq: 0.25
    
    flow:
      transform:
        net:
          channels: [64]
          activation: tanh
      net:
        channels: [4]
        activation: tanh
        kernel_radius: 2
      n_blocks: 4
    
train:
  n_steps: 2000
  batch_size: 10
  init_lr: 0.005
  display_metrics: false

output: false
cuda: true
"""
config = parser.parse_string(config)
print(parser.dump(config))
```

```python
config_str = """
model:
  class_path: torchlft.model_zoo.gaussian.ConvCouplingModel
  init_args:
    target:
      lattice_length: {L}
      lattice_dim: 2
      m_sq: {m_sq}
    
    flow:
      transform:
        scale_fn: exponential
        symmetric: {symmetric}
        shift_only: {shift_only}
      net:
        channels: {channels}
        activation: {activation}
        kernel_radius: {kernel_radius}
        #bias: \{bias\}
      n_blocks: {depth}
      #final_layer_bias: \{bias\}
      #final_diagonal_layer: \{final_diagonal\}
      
    partitioning: {partitioning}

train:
  n_steps: 8000
  batch_size: 4096
  init_lr: 0.005
  display_metrics: false

output: false
cuda: true
"""


def get_config(
    L: int = 8,
    m_sq: float = 0.25,
    channels: list[int] = [8],
    depth: int = 1,
    symmetric: bool = False,
    shift_only: bool = False,
    partitioning: str = "checkerboard",
    final_diagonal: bool = False,
    bias: bool = False,
    activation: str = "tanh",
    kernel_radius: int = 1,
):
    config = config_str.format(
        L=L,
        m_sq=m_sq,
        channels=channels,
        depth=depth,
        symmetric=symmetric,
        shift_only=shift_only,
        partitioning=partitioning,
        # final_diagonal=final_diagonal,
        # bias=bias,
        activation=activation,
        kernel_radius=kernel_radius,
    )
    return parser.parse_string(config)
```

```python
# TEST
config = get_config()
print(parser.dump(config))

instantiated_config = parser.instantiate_classes(config)
untrained_model = instantiated_config.model
_ = untrained_model(1)
print(untrained_model)
```

```python

```
