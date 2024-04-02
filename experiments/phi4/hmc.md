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

# Flowed Hybrid Monte Carlo

```python
import torch
import matplotlib.pyplot as plt

from torchlft.scripts.train import parser, main

from plot import plot_metrics

from torchlft.lattice.sample import DefaultSampler
from torchlft.lattice.scalar.hmc import HamiltonianGaussianMomenta, HybridMonteCarlo
from torchlft.lattice.action import PullbackAction
from torchlft.lattice.scalar.action import Phi4Action

from torchlft.lattice.autocorrelation import compute_autocorrelations


plt.style.use("seaborn-v0_8-paper")

CUDA_AVAILABLE = torch.cuda.is_available()
```

```python
_global_defaults = dict(
    L=8,
    beta=0.576,
    lam=0.5,
    m_sq=0.25,
    n_steps=5000,
    batch_size=4096,
    output="null",
    cuda=CUDA_AVAILABLE,
)
```

```python
_ir_config_str = """
model:
  class_path: torchlft.models.scalar.CouplingModel
  init_args:
    target:
      lattice_length: {L}
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
        radius: {radius}
        n_layers: {n_layers}
      spline:
        transform:
          n_bins: 8
          bounds: 4.0
        point_net:
          channels: {channels}
          activation: tanh
          bias: true
        radius: {radius}
        n_layers: 2
      rescale:
        init_scale: 1.0
        frozen: false

train:
  n_steps: {n_steps}
  batch_size: {batch_size}
  init_lr: 0.001
  display_metrics: true
  log_interval: 1000

output: {output}
cuda: {cuda}
"""

_ir_defaults = dict(
    channels=[64],
    radius=1,
    n_layers=8,
)
   

def _get_ir_config(kwargs):
    kwargs = _global_defaults | _ir_defaults | kwargs
    config = _ir_config_str.format(**kwargs)
    return parser.parse_string(config)
```

```python
_uv_config_str = """
model:
  class_path: torchlft.models.scalar.CouplingModel
  init_args:
    target:
      lattice_length: {L}
      β: {beta}
      λ: {lam}
    flow:
      to_free: null
      affine:
        transform:
          symmetric: true
        point_net:
          channels: {channels}
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
  log_interval: 1000

output: {output}
cuda: {cuda}
"""

_uv_defaults = dict(
    channels=[64],
    radius=1,
    n_layers=8,
)
   

def _get_uv_config(kwargs):
    kwargs = _global_defaults | _uv_defaults | kwargs
    config = _uv_config_str.format(**kwargs)
    return parser.parse_string(config)
```

```python
_linear_config_str = """
model:
  class_path: torchlft.models.scalar.CouplingModel
  init_args:
    target:
      lattice_length: {L}
      β: {beta}
      λ: {lam}
    flow:
      to_free: null
      affine:
        transform:
          symmetric: true
          shift_only: true
        point_net:
          channels: []
          activation: identity
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
  log_interval: 1000

output: {output}
cuda: {cuda}
"""

_linear_defaults = dict(
    channels=[64],
    radius=1,
    n_layers=4,
)
   

def _get_linear_config(kwargs):
    kwargs = _global_defaults | _linear_defaults | kwargs
    config = _linear_config_str.format(**kwargs)
    return parser.parse_string(config)
```

```python
def get_config(model: str, **kwargs):
    if model == "linear":
        return _get_linear_config(kwargs)
    elif model == "uv":
        return _get_uv_config(kwargs)
    elif model == "ir":
        return _get_ir_config(kwargs)
    else:
        print("Model not recognised")
```

## Preliminary check of autocorrelation func

```python
L = 8
beta = 0.576
lam=0.5


linear_model, linear_logger = main(get_config("linear", L=L, beta=beta, lam=lam))
uv_model, uv_logger = main(get_config("uv", L=L, beta=beta, lam=lam))
ir_model, ir_logger = main(get_config("ir", L=L, beta=beta, lam=lam))

vanilla_action = Phi4Action(lattice=(L, L), λ=lam, β=beta)
linear_action = PullbackAction(linear_model)
uv_action = PullbackAction(uv_model)
ir_action = PullbackAction(ir_model)
```

```python
def run_hmc(action, lattice, step_size, n_replica=16, n_traj=5000, n_therm=1000):
    sampler = DefaultSampler(n_traj, n_therm)
    hamiltonian = HamiltonianGaussianMomenta(action)    
    alg = HybridMonteCarlo(lattice, hamiltonian, step_size=step_size, n_replica=n_replica)
    configs, stats = sampler.sample(alg)
    return configs, stats
```

```python
vanilla_configs, vanilla_stats = run_hmc(vanilla_action, [L, L], step_size=0.05)

print(vanilla_stats)

vanilla_autocorr = compute_autocorrelations(vanilla_configs.flatten(start_dim=2).mean(dim=2).T)
print(vanilla_autocorr.integrated)
```

```python
linear_configs, linear_stats = run_hmc(linear_action, [L, L], 0.05)

print(linear_stats)

linear_autocorr = compute_autocorrelations(linear_configs.flatten(start_dim=2).mean(dim=2).T)
print(linear_autocorr.integrated)
```

```python
uv_configs, uv_stats = run_hmc(uv_action, [L, L], 0.01)

print(uv_stats)

uv_autocorr = compute_autocorrelations(uv_configs.flatten(start_dim=2).mean(dim=2).T)
print(uv_autocorr.integrated)
```

```python
ir_configs, ir_stats = run_hmc(ir_action, [L, L], 0.01)

print(ir_stats)

ir_autocorr = compute_autocorrelations(ir_configs.flatten(start_dim=2).mean(dim=2).T)
print(ir_autocorr.integrated)
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

fig, ax = plt.subplots()

ax.plot(vanilla_autocorr.autocorrelation, label="Vanilla HMC")
ax.plot(linear_autocorr.autocorrelation, label="Linear Flow, ESS $= 0.01(1)$")
ax.plot(uv_autocorr.autocorrelation, label="UV Flow, ESS $= 0.10(4)$")
ax.plot(ir_autocorr.autocorrelation, label="IR Flow, ESS $= 0.925(4)$")

ax.annotate(r"$\tau_M \approx 20$", xy=(10, 0.6), xytext=(12, 0.65), arrowprops=dict(width=1, headwidth=5, headlength=5, color="k"))
ax.annotate(r"$\tau_M \approx 6$", xy=(5.1, 0.4), xytext=(6.2, 0.45), arrowprops=dict(width=1, headwidth=5, headlength=5, color="k"))
ax.annotate(r"$\tau_M \approx 2$", xy=(3, 0.19), xytext=(3.6, 0.24), arrowprops=dict(width=1, headwidth=5, headlength=5, color="k"))


ax.set_xscale("log")
ax.set_ylim(0, 1)
ax.set_xlim(1, 100)
ax.legend(ncols=1)
ax.set_xlabel(r"$\delta t$")
ax.set_ylabel(r"$\Gamma_M(\delta t)$")

fig.tight_layout()
fig.savefig("hmc_indicative.png", dpi=300)
```

<!-- #raw -->
ir_configs, ir_stats = run_hmc(ir_action, [L, L], 0.02)

print(ir_stats)

ir_autocorr = compute_autocorrelations(ir_configs.flatten(start_dim=2).mean(dim=2).T)
print(ir_autocorr.integrated)
<!-- #endraw -->

```python
for logger in (linear_logger, uv_logger, ir_logger):
    ess = logger.get_data()["ess"][-1]
    print(f"{ess.mean():.3f} +/- {ess.std():.3f}")
    

fig, ax = plt.subplots()
ax.plot(linear_logger.get_data()["ess"].mean(dim=1))
ax.plot(uv_logger.get_data()["ess"].mean(dim=1))
ax.plot(ir_logger.get_data()["ess"].mean(dim=1))
```

```python

```
