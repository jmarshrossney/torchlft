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

# Free Scalar Fields

```python
import torch
import matplotlib.pyplot as plt

from torchlft.nflow.train import ReverseKLTrainer

from torchlft.scripts.train import parser, main
```

```python
class Trainer(ReverseKLTrainer):
    def logging_step(self, model, step):
        error = model.mask * model.weight - model.cholesky
        rms_error = error.pow(2).sum().sqrt().float()
        abs_max_error = error.abs().max().float()
        print(rms_error, abs_max_error)
        return super().logging_step(model, step)
```

```python
config = {
    "model":
    {
        "class_path": "torchlft.scalar.flows.linear.Model",
        "init_args": {
            "lattice_length": 6,
            "lattice_dim": 2,
            "m_sq": 2,
        },
    },
    "train": {
        "n_steps": 2000,
        "batch_size": 2000,
        "init_lr": 0.005,
    },
    "cuda": False,# torch.cuda.is_available(),
}
config = parser.parse_object(config)
print(parser.dump(config))
```

```python
model, logger, train_dir = main(config)
```

```python
expected_weights = model.cholesky
empirical_weights = model.mask * model.weight.detach()
plt.imshow(expected_weights)# - empirical_weights)
plt.colorbar()
```

```python
print(
    (model.cholesky @ model.cholesky.T - model.covariance).abs().max()
)
```

```python
from torchlft.nflow.utils import get_jacobian

jac, input, output = get_jacobian(model)

print(torch.allclose(jac, model.mask * model.weight))
```

```python
expected_cov = model.covariance

sample, weights = model.weighted_sample(10000)


empirical_cov = torch.cov(sample.transpose(0, 1))
```

```python
_ = plt.hist((expected_cov - empirical_cov).flatten(), bins=25)
```

```python
plt.imshow(expected_cov - empirical_cov)
plt.colorbar()
```

```python
sample, indices = model.metropolis_sample(10000)
indices = indices.tolist()
print("Acceptance: ", len(set(indices)) / len(indices))
```

```python
from torchlft.scalar.observables import two_point_correlator
corr = two_point_correlator(sample.view(-1, 6, 6, 1)).roll((2, 2), (0, 1))
plt.imshow(corr)
plt.colorbar()
```
