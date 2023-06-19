from math import pi as π

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


def ramp(
    x: Tensor, log_scale: Tensor, power: int, eps: float = 1e-9
) -> Tensor:
    α, β = log_scale.exp(), power
    x_masked = torch.where(x > eps, x, torch.full_like(x, eps))
    y = torch.where(
        x > eps,
        torch.exp(-α * x_masked.pow(-β)) / torch.exp(-α),
        torch.zeros_like(x),
    )
    return y

def bump(self, x: Tensor, loga: Tensor, b: Tensor, c: Tensor, logα: Tensor, β: Tensor) -> tuple[Tensor, Tensor]:
    assert (x >= 0).all()
    assert (x <= 1).all()

    a = loga.exp()
    α = logα.exp()

    x10 = torch.stack([x, torch.zeros_like(x), torch.ones_like(x)])

    h_x10 = a * (x10 - b) + (1 / 2)  # affine transform of x

    ρ_x10 = ramp(h_x10, logα, β)
    ρ1m_x10 = ramp(1 - h_x10, logα, β)

    σ_x10 = ρ_x10 / (ρ_x10 + ρ1m_x10)

    σx, σ0, σ1 = σ_x10

    y = c * (σx - σ0) / (σ1 - σ0) + (1 - c) * x

    dσdx = (
        a
        * σx
        * (1 - σx)
        * (β / α)
        * ((x.pow(β + 1) + (1 - x).pow(β + 1)) / (x * (1 - x)).pow(β + 1))
    )

    dydx = c * dσdx / (σ1 - σ0) + (1 - c)

    return y, dydx

class BumpTransform(nn.Module):
    flow = None
    def __init__(self):
        super().__init__(even: bool, circular: bool = False)

        if circular:
            def bump_(self, x: Tensor, mask: LongTensor, loga: Tensor, b: Tensor, c: Tensor, logα: Tensor, β: Tensor):
                x, loga, b, c, logα = x * mask, loga * mask, b * mask, c * mask, logα * mask
                y, dydx = bump(x / (2 * π), loga=loga, b=b, c=c, logα=logα, β=β)
                return (y * mask) * (2 * π), (dydx.log() * mask).flatten(start_dim=1).sum(dim=1)
        else:
            def transform(x, **kwargs):
                x, loga, b, c, logα = x * mask, loga * mask, b * mask, c * mask, logα * mask
                y, dydx = bump((x + 1) / 2, loga=loga, b=b, c=c, logα=logα, β=β)
                return y * 2 - 1, dydx.log().flatten(start_dim=1).sum(dim=1)
                x = ((x + 1) * mask) / 2, loga * mask, b * mask, c * mask, logα * mask
                y, dydx = bump(x / (2 * π), loga=loga, b=b, c=c, logα=logα, β=β)
                return (y * mask) * (2 * π), (dydx.log() * mask).flatten(start_dim=1).sum(dim=1)
        self.transform = transform

        self.even = even


    def forward(self, x_passive: Tensor, y_active: Tensor | None):

        condit = x_passive if y_active is None else torch.cat([x_passive, y_active], dim=1)
        mask = self.flow.checker if self.even else (1 - self.flow.checker)

        params = self.net(condit) * mask

        loga, b, c, logα = params.split(1, dim=1)
        β = 2

        b = torch.sigmoid(b)
        c = torch.sigmoid(c)

        return partial(self.transform, loga=loga, b=b, c=c, logα=logα, β=β)

def make_cnn(channels_in: int, hidden_shape: list[int], kernel_radius: int) -> nn.Sequential:
    kernel_size = kernel_radius * 2 + 1
    layers = [
        nn.Conv2d(
            channels_in,
            channels_out,
            kernel_size,
            padding=kernel_size // 2,
            padding_mode="circular",
            bias=False,
        )
        for channels_in, channels_out in zip(
            [channels_in, *hidden_shape], [*hidden_shape, 4]
        )
    ]
    activations = [nn.Tanh() for _ in hidden_shape] + [nn.Tanh()]
    return nn.Sequential(*list(chain(*zip(layers, activations))))


def make_checkerboard(x: int) -> torch.LongTensor:
    checker = x.new_zeros(x.shape[1:], dtype=torch.long)
    assert checker.dim() == 2
    checker[::2, ::2] = 1
    checker[1::2, 1::2] = 1
    return checker


class RecursiveFlow(nn.Module):
    def __init__(
        self,
        ordering: list[int],
        epsilon: float = 1e-3,
    ):
        super().__init__()

        D = len(interval_transforms) + 1
        assert D > 1
        assert len(ordering) == D + 1
        assert max(ordering) == D + 1
        assert min(ordering) == 1

        self.interval_transforms = nn.ModuleList(BumpTransform() for _ in range(D - 1))
        self.circular_transform = BumpTransform(circular=True)

        self.ordering = [i - 1 for i in ordering]  # zero indexing
        self.inverse_ordering = sorted(
            range(D + 1), key=self.ordering.__getitem__
        )

        self._dim = D
        self.epsilon = epsilon

    def new_lattice_(self) -> None:
        def _register_checkerboard(self_, inputs: tuple[Tensor]):
            (x,) = inputs
            checkerboard = make_checkerboard(x)
            self_.register_buffer("checker", checkerboard)
            self_._checkerboard_hook.remove()

        self._checkerboard_hook = self.register_forward_pre_hook(
            _register_checkerboard
        )

    @property
    def dim(self) -> int:
        return self._dim

    def _safe_mask(self, x: Tensor) -> BoolTensor:
        return (1 - x**2) > self.epsilon**2

    def forward_(self, x: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        assert x.dim() == 2
        assert x.shape[1] == self.dim + 1

        D = self.dim
        ε = self.epsilon

        ldj_total = x.new_zeros(x.shape[0])

        x = x[:, self.ordering]

        # Unwrap onto cylinder, counting down from i=D to i=2
        x_sphere = x
        x_intervals = []
        scale_factors = []
        for i in range(D, 1, -1):
            x_i, x_sphere = x_sphere.tensor_split([1], dim=1)
            x_intervals.append(x_i)

            safe_mask = self._safe_mask(x_i)
            ρ_i = torch.where(safe_mask, 1 - x_i**2, ε**2).sqrt()
            assert ρ_i.isfinite().all()

            x_sphere = torch.where(safe_mask, x_sphere / ρ_i, x_sphere / ε)

            scale_factors.append(ρ_i)

        # Should just have circular components remaining
        assert x_sphere.shape[1] == 2
        x_circle = x_sphere

        zip_x_and_transform = zip(x_intervals, self.interval_transforms)

        # Unconditionally transform the first component
        x1, f1 = next(zip_x_and_transform)
        y1, ldj_1 = f1(k)(x1)
        ldj_total += ldj_1

        # Transform remaining, conditioned on those already transformed
        y_intervals = y1
        for x_i, f_i in zip_x_and_transform:
            y_i, ldj_i = f_i(k, y_intervals)(x_i)
            ldj_total += ldj_i

            y_intervals = torch.cat([y_intervals, y_i], dim=1)

        # Transform circular part, conditioned on all interval parts
        x_D = as_angle(x_circle)
        y_D, ldj_D = self.circular_transform(k, y_intervals)(x_D)
        ldj_total += ldj_D
        y_circle = as_vector(y_D + self.rotation)

        # Wrap back onto the sphere, counting up from i=2 to i=D
        y_sphere = y_circle
        scale_factors_inverse = []
        for i in range(2, D + 1, +1):
            y_intervals, y_i = y_intervals.tensor_split([-1], dim=1)

            safe_mask = self._safe_mask(y_i)
            r_i = torch.where(safe_mask, 1 - y_i**2, ε**2).sqrt()
            assert r_i.isfinite().all()

            y_sphere = torch.where(safe_mask, y_sphere * r_i, y_sphere * ε)

            y_sphere = torch.cat([y_i, y_sphere], dim=1)  # prepended!
            scale_factors_inverse.insert(0, r_i)  # prepended!

        assert y_intervals.numel() == 0
        assert y_sphere.shape[1] == self.dim + 1

        # reorder
        y = y_sphere[:, self.inverse_ordering]

        # Compute ldj for the cylinder->sphere transformation and inverse
        # Take advantage of cancellation of large ρ and r near the poles
        for D_i, ρ_i, r_i in zip(
            range(D, 1, -1),
            scale_factors,
            scale_factors_inverse,
            strict=True,
        ):
            ldj_total -= (D_i - 2) * torch.log(ρ_i / r_i).squeeze(1)

        return y, ldj_total

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x_even, x_odd = x * self.checker, x * (1 - self.checker)

        y_even, ldj_even = self.forward_(x_even, x_odd)
        assert torch.allclose(y_even, y_even * self.checker)

        y_odd, ldj_odd = self.forward(x_odd, y_even)
        assert torch.allclose(y_odd, y_odd * self.checker)

        return y_even + y_odd, ldj_even + ldj_odd


