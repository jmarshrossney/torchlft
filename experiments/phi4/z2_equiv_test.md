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

# Z2 Equivariance testing

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

```python
_config_str = """
model:
  class_path: torchlft.models.scalar.DenseCouplingModel
  init_args:
    target:
      lattice_length: {L}
      β: {beta}
      λ: {lam}
    flow:
      to_free: null
      affine:
        transform:
          symmetric: {symmetric}
        net:
          sizes: {sizes}
          activation: tanh
          bias: {bias}
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

_defaults = dict(
    L=8,
    beta=0.576,
    lam=0.5,
    m_sq=0.25,
    sizes=[128],
    n_layers=8,
    n_steps=4000,
    batch_size=4096,
    init_lr=0.001,
    cuda=CUDA_AVAILABLE,
)

def get_config(**kwargs):
    kwargs = _defaults | kwargs
    config = _config_str.format(**kwargs)
    return parser.parse_string(config)
```

```python
magnetisations = []
acceptances = []
effective_sample_sizes = []

for beta in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
    for symmetric in [False, True]:
        config = get_config(beta=beta, symmetric=symmetric, bias=not symmetric)
        trained_model, logger = main(config)

        configs, _ = trained_model.weighted_sample(10000)
        mag = configs.mean(dim=-1)
        mag_mean = mag.mean()
        mag_std = mag.std()
        magnetisations.append([mag_mean, mag_std])
        
        data = logger.get_data()
        ess = data["ess"][-1]
        acc = data["acc"][-1].type_as(ess)
        q = torch.tensor([0.0, 0.5, 1.0]).type_as(ess)

        acceptances.append(acc.quantile(q))
        effective_sample_sizes.append(ess.quantile(q))

        print(f"beta: {beta} , symmetric: {symmetric} , mag: {float(mag_mean)} , Final acceptance: {acc[1]}")

        #fig, ax = plt.subplots()
        #ax.hist(configs.flatten(), bins=25)
        

```

<!-- #raw -->
fig, ax = plt.subplots()

ax.plot(acceptances, label="acc")
ax.plot(effective_sample_sizes, label="ess")

ax.legend()

<!-- #endraw -->

```python
beta = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
m = torch.tensor(magnetisations)
a = torch.stack(acceptances)
e = torch.stack(effective_sample_sizes)
m.shape, a.shape, e.shape
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

fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={"hspace": 0, "height_ratios": [2, 1]})

ax.plot(beta, a[::2, 0], "o:", label="Standard")
ax.plot(beta, a[1::2, 0], "o:", label=r"$\mathbb{Z}_2$ equivariant")
ax.set_ylabel("Acceptance")

ax2.plot(beta, m[::2, 0].abs(), "o:")
ax2.plot(beta, m[1::2, 0].abs(), "o:")
ax2.set_xlabel(r"$\beta$")
ax2.set_ylabel(r"$|\langle \phi \rangle |$")

ax.legend()
fig.tight_layout()

fig.savefig("z2_test.png", dpi=300)
```

```python

```
