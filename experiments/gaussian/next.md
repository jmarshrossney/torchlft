# Coupling Flow for Free Scalar Theory in $d=2$

```python
import torch
import matplotlib.pyplot as plt

from torchlft.scripts.train import parser, main

plt.style.use("seaborn-v0_8-notebook")
```

## Coupling layers with dense networks

```python
config = """
model:
  class_path: torchlft.model_zoo.gaussian.DenseCouplingModel
  init_args:
    target:
      lattice_length: 6
      lattice_dim: 2
      m_sq: 0.25
    
    flow:
      net:
        sizes: [64]
        activation: tanh
      n_blocks: 2
      
    partitioning: checkerboard

train:
  n_steps: 2000
  batch_size: 2000
  init_lr: 0.005
  display_metrics: false

output: false
cuda: true
"""
config = parser.parse_string(config)
print(parser.dump(config))
```

```python
model, logger = main(config)
```

```python
metrics = logger.get_data()

print([(k, v.shape) for k, v in metrics.items()])

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
```

```python
expected_cov = model.target.covariance
expected_var = expected_cov[0, 0]

sample, weights = model.weighted_sample(10000)

empirical_cov = torch.cov(sample.transpose(0, 1))

_ = plt.hist((expected_cov - empirical_cov).flatten(), bins=25)
```

```python
print(sample.shape)

var = sample.var(dim=0)

plt.imshow(var.view(6, 6))
plt.colorbar()

print(var.var())
```

## Coupling layers with convolutional networks

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
          activation: identity #tanh
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
model, logger = main(config)
```

```python
metrics = logger.get_data()

print([(k, v.shape) for k, v in metrics.items()])

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
```

## Best of both worlds?

```python
import torch.nn as nn

conv = nn.Conv2d(1, 64, kernel_size=3, bias=False)
print([t.numel() for t in conv.parameters()])
print(9 * 64)
```

```python
checker = checkerboard_mask([k + 1, k + 1], offset=1)[:-1, :-1]
print(checker)

a = torch.arange(k**2).view(k, k) * checker
print(a)

a = torch.zeros(k, k, dtype=torch.long).masked_scatter(
    checker, torch.arange(1, k**2 // 2 + 1)
)
# a[checker] = torch.arange(1, k**2 // 2 + 1)
print(a)

one_hot = torch.nn.functional.one_hot(a)

print(one_hot.shape)

print(one_hot[..., 4])
```

```python
import torch
import matplotlib.pyplot as plt

from torchlft.scripts.train import parser, main

plt.style.use("seaborn-v0_8-notebook")
```

```python
config = """
model:
  class_path: torchlft.model_zoo.gaussian.CouplingModel
  init_args:
    target:
      lattice_length: 6
      lattice_dim: 2
      m_sq: 0.25
    
    flow:
      transform:
        net:
          channels: [64]
          activation: tanh
      radius: 2
      n_blocks: 2
    
train:
  n_steps: 2000
  batch_size: 2000
  init_lr: 0.005
  display_metrics: false

output: false
cuda: true
"""
config = parser.parse_string(config)
print(parser.dump(config))
```

```python
model, logger = main(config)
```

```python
metrics = logger.get_data()

print([(k, v.shape) for k, v in metrics.items()])

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
```

## Z2 equivariance

```python

```

## Receptive field and correlation length

```python

```

```python

```
