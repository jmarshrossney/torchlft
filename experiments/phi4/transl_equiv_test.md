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

# Translation Equivariance Test

```python
import torch
import matplotlib.pyplot as plt

from torchlft.scripts.train import parser, main
from torchlft.nflow.utils import get_jacobian, get_model_jacobian
from torchlft.nflow.layer import Layer

from plot import (
    plot_metrics,
    plot_layer_jacobians,
    plot_layer_log_det_jacobians,
)

plt.style.use("seaborn-v0_8-paper")

CUDA_AVAILABLE = torch.cuda.is_available()
```

## Config

```python
_global_defaults = dict(
    L=8,
    beta=0.576,
    lam=0.5,
    m_sq=0.25,
    n_layers=8,
    n_steps=4000,
    batch_size=4096,
    init_lr=0.001,
    cuda=CUDA_AVAILABLE,
)
```

```python
_linear_config_str = """
model:
  class_path: torchlft.models.scalar.DenseCouplingModel
  init_args:
    target:
      lattice_length: {L}
      β: {beta}
      λ: {lam}
    flow:
      to_free:
        m_sq: {m_sq}
        frozen: false
      affine: null
      spline: null
      rescale: null

    partitioning: checkerboard

train:
  n_steps: {n_steps}
  batch_size: {batch_size}
  init_lr: {init_lr}
  display_metrics: false
  print_model_summary: false

output: false
cuda: {cuda}
"""

def _get_linear_config(kwargs):
    kwargs = _global_defaults | kwargs
    config = _linear_config_str.format(**kwargs)
    return parser.parse_string(config)
```

```python
_dense_config_str = """
model:
  class_path: torchlft.models.scalar.DenseCouplingModel
  init_args:
    target:
      lattice_length: {L}
      #m_sq: {m_sq}
      β: {beta}
      λ: {lam}
    flow:
      to_free:
        m_sq: {m_sq}
      affine:
        transform:
          symmetric: true
        net:
          sizes: {sizes}
          activation: tanh
          bias: false
        n_layers: {n_layers}
      spline: null
      rescale:
        init_scale: 1.0
        frozen: false

    partitioning: checkerboard

train:
  n_steps: {n_steps}
  batch_size: {batch_size}
  init_lr: {init_lr}
  display_metrics: false
  print_model_summary: false

output: false
cuda: {cuda}
"""

_dense_defaults = dict(
    sizes=[128],
)
   

def _get_dense_config(kwargs):
    kwargs = _global_defaults | _dense_defaults | kwargs
    config = _dense_config_str.format(**kwargs)
    return parser.parse_string(config)
```

```python
_conv_config_str = """
model:
  class_path: torchlft.models.scalar.ConvCouplingModel
  init_args:
    target:
      lattice_length: {L}
      #m_sq: {m_sq}
      β: {beta}
      λ: {lam}
    flow:
      to_free:
        m_sq: {m_sq}
      affine:
        transform:
          symmetric: true
        point_net:
          channels: {channels}
          activation: tanh
          bias: false
        spatial_net:
          channels: {sizes}
          activation: tanh
          bias: false
          kernel_radius: {kernel_radius}
        n_layers: {n_layers}
      spline: null
      rescale:
        init_scale: 1.0
        frozen: false

train:
  n_steps: {n_steps}
  batch_size: {batch_size}
  init_lr: {init_lr}
  display_metrics: false
  print_model_summary: false

output: false
cuda: {cuda}
"""

_conv_defaults = dict(
    channels=[4],
    sizes=[128],
    kernel_radius=1,
)

def _get_conv_config(kwargs):
    kwargs = _global_defaults | _conv_defaults | kwargs
    config = _conv_config_str.format(**kwargs)
    return parser.parse_string(config)
```

```python
_equivar_config_str = """
model:
  class_path: torchlft.models.scalar.CouplingModel
  init_args:
    target:
      lattice_length: {L}
      #m_sq: {m_sq}
      β: {beta}
      λ: {lam}
    flow:
      to_free:
        m_sq: {m_sq}
      affine:
        transform:
          symmetric: true
        point_net:
          channels: {sizes}
          activation: tanh
          bias: false
        radius: {radius}
        n_layers: {n_layers}
      spline: null
      rescale:
        init_scale: 1.0
        frozen: false

train:
  n_steps: {n_steps}
  batch_size: {batch_size}
  init_lr: 0.001
  display_metrics: false
  print_model_summary: false

output: false
cuda: {cuda}
"""

_equivar_defaults = dict(
    sizes=[128],
    radius=1,
)
  

def _get_equivar_config(kwargs):
    kwargs = _global_defaults | _equivar_defaults | kwargs
    config = _equivar_config_str.format(**kwargs)
    return parser.parse_string(config)
```

```python
def get_config(model: str, **kwargs):
    if model == "linear":
        return _get_linear_config(kwargs)
    elif model == "dense":
        return _get_dense_config(kwargs)
    elif model == "conv":
        return _get_conv_config(kwargs)
    elif model == "equivar":
        return _get_equivar_config(kwargs)
    else:
        print("Model not recognised")
```

<!-- #raw -->
# TEST
for model in ("linear", "dense", "conv", "equivar"):
    
    config = get_config(model)
    print(parser.dump(config))

    instantiated_config = parser.instantiate_classes(config)
    untrained_model = instantiated_config.model
    _ = untrained_model(1)
    print(untrained_model)
<!-- #endraw -->

## Plot

```python
config_dict = {}

config_dict["dense"] = dict(model="dense", sizes=[128])
config_dict["big_conv"] = dict(model="conv", sizes=[128], channels=[16])
config_dict["small_conv"] = dict(model="conv", sizes=[32], channels=[4])
config_dict["big_single"] = dict(model="equivar", sizes=[128])
config_dict["small_single"] = dict(model="equivar", sizes=[32])

results = {}

for label, config in config_dict.items():
    config = get_config(**config)
    trained_model, logger = main(config)

    acceptance = float(logger.get_data()["acc"].mean(dim=1)[-1])
    print(f"{label} (P = {trained_model.parameter_count}) --- Final acceptance: {acceptance}")
    
    results[label + "_from_free"] = logger.data[4000] | {"n_params": trained_model.parameter_count}

    config["model.init_args.flow.to_free"] = None
    trained_model, logger = main(config)

    acceptance = float(logger.get_data()["acc"].mean(dim=1)[-1])
    print(f"{label} model (P = {trained_model.parameter_count}) --- Final acceptance: {acceptance}")

    results[label + "_from_iso"] = logger.data[4000] | {"n_params": trained_model.parameter_count}
    
```

```python
labels = []
acceptances = []
eff_sample_sizes = []
params = []

for label, res in results.items():
    labels.append(label)

    params.append(res["n_params"])
    acceptances.append(res["acc"].float().quantile(torch.tensor([0.0, 0.5, 1.0])))
    eff_sample_sizes.append(res["ess"].float().quantile(torch.tensor([0.0, 0.5, 1.0])))

acceptances = torch.stack(acceptances)
eff_sample_sizes = torch.stack(eff_sample_sizes)
```

```python
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


fig, ax = plt.subplots(layout="constrained")
ax.set_ylim(0, 0.75)
print(labels)

l1 = labels[::2]
l2 = labels[1::2]
p1 = params[::2]
p2 = params[1::2]

x = torch.arange(len(labels) // 2)
y1, y2 = acceptances[:, 1].view(-1, 2).T
yerr1, yerr2 = (acceptances[:, ::2] - acceptances[:, 1:2]).abs().view(-1, 2, 2).transpose(1, 0)

b1 = ax.bar(x - 0.2, y2, yerr=yerr2.T, width=0.4, linewidth=1, edgecolor="k", error_kw=dict(elinewidth=1, capsize=4, capthick=1))
b2 = ax.bar(x + 0.2, y1, yerr=yerr1.T, width=0.4, linewidth=1, edgecolor="k", capsize=10, error_kw=dict(elinewidth=1, capsize=4, capthick=1))

ax.bar_label(b1, labels=p1)
ax.bar_label(b2, labels=p2)

ax.set_xticks(x, ["Dense", "Conv (big)", "Conv (small)", "Single (big)", "Single (small)"])
ax.set_xlabel("Conditioner Network")
ax.set_ylabel("Acceptance")

#ax.legend(handles=[b1, b2], labels=["From $\mathcal{N}(0, I)$", "From $\mathcal{N}(0, K^{-1})$"])
ax.legend(handles=[b1, b2], labels=["No initial layer", "Initial Cholesky layer"], ncols=2, loc="upper left")

ax.annotate("num. params", xy=(4.1, 0.63), xytext=(3.6, 0.68), arrowprops=dict(width=1, headwidth=5, headlength=5, color="k"))

fig.savefig("trans_equiv.png", dpi=300)
```
