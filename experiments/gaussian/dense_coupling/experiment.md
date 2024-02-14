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

# Dense Coupling Flows and Gaussian Fields

```python
import torch
import matplotlib.pyplot as plt

from torchlft.scripts.train import parser, main
from torchlft.nflow.utils import get_jacobian, get_model_jacobian
from torchlft.nflow.layer import Layer

plt.style.use("seaborn-v0_8-notebook")
```

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
        activation: tanh
      n_blocks: {depth}
      
    partitioning: {partitioning}

train:
  n_steps: 2000
  batch_size: 1024
  init_lr: 0.005
  display_metrics: false

output: false
cuda: true
"""

def get_config(
    L: int,
    m_sq: float,
    sizes: list[int] = [64],
    depth: int = 1,
    symmetric: bool = False,
    shift_only: bool = False,
    partitioning: str = "checkerboard",
):
    config = config_str.format(
        L=L,
        m_sq=m_sq,
        sizes=sizes,
        depth=depth,
        symmetric=symmetric,
        shift_only=shift_only,
        partitioning=partitioning,
    )
    return parser.parse_string(config)
```

```python
# TEST
config = get_config(L=8, m_sq=0.25, depth=2, partitioning="lexicographic")
print(parser.dump(config))

instantiated_config = parser.instantiate_classes(config)
untrained_model = instantiated_config.model
_ = untrained_model(1)
print(untrained_model)
```

## Jacobian

```python
def plot_layer_jacobians(model_):
    layers = [(name, mod) for name, mod in model_.named_modules() if isinstance(mod, Layer)]
    #print(layers)
    
    flow, *layers = layers
    layers = layers + [flow, ("full model", model_.flow_forward)]
    
    inputs, _ = model_.sample_base(64)

    n_rows = len(layers) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(8, 4 * n_rows))
    axes = iter(axes.flatten())
        
    for label, transform in layers:
        
        with torch.no_grad():
            jac, _, _ = get_jacobian(transform, inputs)
    
        ax = next(axes)
        ax.imshow(jac.pow(2).sum(0).sqrt())
        ax.set_title(f"{label}")
    
    fig.tight_layout()

    return fig
```

```python
_ = plot_layer_jacobians(untrained_model)
```

```python
trained_model, logger = main(config)

acceptance = float(logger.get_data()["acc"].mean(dim=1)[-1])
print(f"Final acceptance: {acceptance}")

_ = plot_layer_jacobians(trained_model)
```

## Log Det Jacobian

```python
config = get_config(L=8, m_sq=0.25, depth=8, partitioning="checkerboard")
print(parser.dump(config))

instantiated_config = parser.instantiate_classes(config)
untrained_model = instantiated_config.model
_ = untrained_model(1)
print(untrained_model)

trained_model, logger = main(config)

acceptance = float(logger.get_data()["acc"].mean(dim=1)[-1])
print(f"Final acceptance: {acceptance}")
```

```python
def plot_layer_ldj(model_):
    layers = [(name, mod) for name, mod in model_.named_modules() if isinstance(mod, Layer)]
    #print(layers)
    
    flow, *layers = layers
    layers = layers# + [flow, ("full model", model_.flow_forward)]
    
    inputs, _ = model_.sample_base(64)

    fig, ax = plt.subplots()

    with torch.no_grad():
        data = [transform(inputs)[1] for _, transform in layers]

    data = torch.cat(data, dim=-1)
    print(data.shape)
    
    ax.fill_between(range(data.shape[1]), data.quantile(.25, dim=0), data.quantile(.85, dim=0), alpha=0.6)
    ax.plot(range(data.shape[1]), data.quantile(.5, dim=0), "--")
            
    fig.tight_layout()

    return fig
```

```python
_ = plot_layer_ldj(untrained_model)
_ = plot_layer_ldj(trained_model)
```

## Smee

```python
def plot_metrics(logger):
    metrics = logger.get_data()
      
    steps = metrics["steps"]
    kl_div = -metrics["mlw"]
    one_minus_ess = 1 - metrics["ess"]
    one_minus_acc = 1 - metrics["acc"]
    vlw = metrics["vlw"]
    
    print("Mean acceptance: " , float(metrics["acc"].mean(1)[-1]))
    
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

```python
from torchlft.nflow.utils import get_model_jacobian

def plot_jacobian_spectrum(model):
    with torch.no_grad():
        jac, _, _ = get_model_jacobian(model, 64)
        
    λ = torch.linalg.eigvals(jac)
    _, D = λ.shape

    print(λ.imag.abs().max())
    λ_compl = λ.clone()
    
    λ = λ.real
    λ, _ = λ.sort(dim=1)

    fig, axes = plt.subplots(1, 2)
    axes = iter(axes.flatten())

    ax = next(axes)
    ax.fill_between(range(D), λ.quantile(0., dim=0), λ.quantile(1., dim=0), alpha=0.7)
    
    #ax.plot(range(D), λ.quantile(0.5, dim=0), "r:")

    cholesky = model.target.cholesky
    λ_expec = torch.linalg.eigvals(cholesky).real
    λ_expec, _ = λ_expec.sort()
    ax.plot(range(D), λ_expec, "k-", label="Cholesky")
    print(λ_expec)

    ax.legend()

    ax = next(axes)
    det_expec = torch.linalg.det(cholesky)
    det = torch.linalg.det(jac)
    #ax.hist(det.log(), label="model")
    ax.axvline(det_expec.log(), label="Choleksy", color="r")

    eig_prod = torch.prod(λ_compl, dim=1)
    ax.hist(eig_prod.real.log(), label="Real product of eigs")
    max_imag_log_det = eig_prod.imag.clamp(0).log().max()
    print(f"Max imaginary part: {max_imag_log_det}")
    
    ax.legend()
    
    return fig
```

## Jacobian

```python
config = get_config(L=8, m_sq=0.25, depth=1, partitioning="lexicographic")

model, logger = main(config)

_ = plot_jacobian(model)
```

## Comparison with Cholesky

```python
config = get_config(L=8, m_sq=0.25, depth=1)

print(parser.dump(config, skip_none=False))

model, logger = main(config)
```

```python
_ = plot_metrics(logger)
```

```python
_ = plot_jacobian_spectrum(model)
```

```python

```
