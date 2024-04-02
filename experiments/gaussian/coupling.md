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

from plot import (
    plot_metrics,
    plot_model_jacobian_vs_covariance,
    plot_layer_jacobians,
    plot_layer_log_det_jacobians,
    plot_jacobian_qr,
)

plt.style.use("seaborn-v0_8-paper")

CUDA_AVAILABLE = torch.cuda.is_available()
```

## Configuration

```python
_global_defaults = dict(
    L=8,
    m_sq=0.25,
    n_layers=4,
    n_steps=2000,
    batch_size=1024,
    init_lr=0.005,
    cuda=CUDA_AVAILABLE,
)
```

```python
_LCM_config_str = """
model:
  class_path: torchlft.models.gaussian.LinearCouplingModel
  init_args:
    target:
      lattice_length: {L}
      lattice_dim: 2
      m_sq: {m_sq}
    flow:
      n_layers: {n_layers}
    partitioning: {partitioning}

train:
  n_steps: {n_steps}
  batch_size: {batch_size}
  init_lr: {init_lr}
  display_metrics: false

output: false
cuda: {cuda}
"""

_LCM_defaults = dict(
    partitioning="lexicographic",
)


def _get_LCM_config(kwargs):
    kwargs = _global_defaults | _LCM_defaults | kwargs
    config = _LCM_config_str.format(**kwargs)
    return parser.parse_string(config)
```

```python
_NLCM_config_str = """
model:
  class_path: torchlft.models.gaussian.NonLinearCouplingModel
  init_args:
    target:
      lattice_length: {L}
      lattice_dim: 2
      m_sq: {m_sq}
    flow:
      net:
        sizes: {sizes}
        activation: {activation}
      n_layers: {n_layers}
      shift_only: {shift_only}
    partitioning: {partitioning}

train:
  n_steps: {n_steps}
  batch_size: {batch_size}
  init_lr: {init_lr}
  display_metrics: false

output: false
cuda: {cuda}
"""

_NLCM_defaults = dict(
    partitioning="lexicographic",
    sizes=[64],
    activation="tanh",
    shift_only=False,
)


def _get_NLCM_config(kwargs):
    kwargs = _global_defaults | _NLCM_defaults | kwargs
    config = _NLCM_config_str.format(**kwargs)
    return parser.parse_string(config)
```

```python
_ELCM_config_str = """
model:
  class_path: torchlft.models.gaussian.EquivLinearCouplingModel
  init_args:
    target:
      lattice_length: {L}
      lattice_dim: 2
      m_sq: {m_sq}
    flow:
      n_layers: {n_layers}
      radius: {radius}

train:
  n_steps: {n_steps}
  batch_size: {batch_size}
  init_lr: {init_lr}
  display_metrics: false

output: false
cuda: {cuda}
"""

_ELCM_defaults = dict(
    sizes=[64],
    radius=1,
)


def _get_ELCM_config(kwargs):
    kwargs = _global_defaults | _ELCM_defaults | kwargs
    config = _ELCM_config_str.format(**kwargs)
    return parser.parse_string(config)
```

```python
def get_config(model: str, **kwargs):
    if model == "linear":
        return _get_LCM_config(kwargs)
    elif model == "nonlinear":
        return _get_NLCM_config(kwargs)
    elif model == "equivar":
        return _get_ELCM_config(kwargs)
    else:
        print("Model not recognised")


_ = get_config("linear")
_ = get_config("nonlinear")
_ = get_config("equivar")
```

```python
# TEST
config = get_config("linear")
print(parser.dump(config))

instantiated_config = parser.instantiate_classes(config)
untrained_model = instantiated_config.model
_ = untrained_model(1)
print(untrained_model)
```

## Linear Model

```python
config = get_config("linear", n_layers=3)

instantiated_config = parser.instantiate_classes(config)
untrained_model = instantiated_config.model
_ = untrained_model(1)
print(untrained_model)
```

```python
trained_model, logger = main(config)

acceptance = float(logger.get_data()["acc"].mean(dim=1)[-1])
print(f"Final acceptance: {acceptance}")

_ = plot_metrics(logger)
```

```python
[fig for fig in plot_layer_jacobians(trained_model)]
```

```python
[fig for fig in plot_model_jacobian_vs_covariance(trained_model)]
```

## Non-linear Model

```python
config = get_config("nonlinear", activation="leaky_relu", init_lr=0.001)

trained_model, logger = main(config)

acceptance = float(logger.get_data()["acc"].mean(dim=1)[-1])
print(f"Final acceptance: {acceptance}")

_ = plot_metrics(logger)
```

```python
_ = plot_layer_log_det_jacobians(trained_model)
```

<!-- #raw -->
_ = plot_jacobian_qr(trained_model)
<!-- #endraw -->

## Equivariant Linear Model

```python
config = get_config("equivar")

instantiated_config = parser.instantiate_classes(config)
untrained_model = instantiated_config.model
_ = untrained_model(1)

z, _ = untrained_model.sample_base(1)
φ1, _ = untrained_model.flow_forward(z)

# Only equivariant under even-valued shifts
δx, δy = 2, -4
μ, ν = 1, 2

φ2, _ = untrained_model.flow_forward(z.roll((δx, δy), (μ, ν)))
φ2 = φ2.roll((-δx, -δy), (μ, ν))

assert torch.allclose(φ1, φ2)
```

```python
config = get_config("equivar", radius=3, n_layers=4)

trained_model, logger = main(config)

acceptance = float(logger.get_data()["acc"].mean(dim=1)[-1])
print(f"Final acceptance: {acceptance}")

_ = plot_metrics(logger)
```

```python
[fig for fig in plot_layer_jacobians(trained_model)]
```

```python
[fig for fig in plot_model_jacobian_vs_covariance(trained_model)]
```

## Evolution of log-det-Jacobian for NonLinear Coupling Model

We see an overshoot for deep models, even for relatively small learning rates...

```python
configs = {}

configs["2x Linear Coupling + Diagonal"] = get_config(
    "linear",
    n_layers=2,
)

for n_layers in (4, 8):#, 12, 16):
    configs[f"{n_layers}x Non-linear Coupling"] = get_config(
        "nonlinear",
        n_layers=n_layers,
        #init_lr=1e-3,
        #n_steps=10000,
    )


models = {}

for label, config in configs.items():

    trained_model, logger = main(config)

    ess = float(logger.get_data()["ess"].mean(dim=1)[-1])
    label += f" (ESS: {ess:.2f})"

    models[label] = trained_model

fig = plot_layer_log_det_jacobians(models)
```

```python
fig.savefig("depth_convergence_smaller_lr.png", dpi=300)
```

## Does the Linear Coupling Model learn the Cholesky decomposition?

...up to an orthogonal transformation?

Doesn't seem like it!

```python
models = {}

# for steps in (100, 500, 1000, 5000):
for n_layers in (4, 8, 12):
    config = get_config(
        "linear",
        L=16,
        m_sq=1 / 16,
        n_layers=n_layers,
        steps=4000,
        batch_size=4096,
    )
    trained_model, logger = main(config)
    mlw = float(logger.get_data()["mlw"].mean(dim=1)[-1])

    # label = f"{steps} steps, (KL: {-mlw:.2g})"
    label = f"{n_layers}x Linear Coupling, (KL: {-mlw:.2g})"
    models[label] = trained_model

fig = plot_jacobian_qr(models)
```

```python

```
