class ConditionalTransform(torch.nn.Module):
    def conditioner(self, x: Tensor) -> Tensor:
        ...

    @property
    def transform(self) -> Callable[[Tensor, ...], Transform]:
        ...

    def forward(self, x: Tensor, context: dict = {}) -> Transform:
        self.context = context
        θ = self.conditioner(x)
        return self.transform(θ)


class FNNConditionalTransform(ConditionalTransform):
    def __init__(
        self,
        size_in: int,
        size_out: int,
        hidden_shape: list[int],
        activation: Activation,
        final_activation: Activation | None = None,
        bias: bool = True,
    ):
        super().__init__()

        layers = [
            nn.Linear(f_in, f_out, bias)
            for f_in, f_out in zip(
                [size_in, *hidden_shape], [*hidden_shape, size_out]
            )
        ]
        activations = [activation for _ in hidden_shape]
        if final_activation is not None:
            activations.append(final_activation)

        self.network = nn.Sequential(*list(chain(*zip(layers, activations))))


class ConditionalTranslation(nn.Module):
    def __init__(self, lattice_shape: list[int], **net_kwargs):
        super().__init__()
        n_lattice = math.prod(lattice_shape)
        self.net = make_fnn(
            size_in=n_lattice, size_out=n_lattice, **net_kwargs
        )

    def forward(self, x: Tensor) -> Transform:
        t = self.net(x)
        return Translation(t)


class ConditionalAffineTransformFNN(nn.Module):
    def __init__(self, lattice_shape: list[int], **net_kwargs):
        super().__init__()
        n_lattice = math.prod(lattice_shape)
        self.net = make_fnn(
            size_in=n_lattice, size_out=n_lattice * 2, **net_kwargs
        )

    def forward(self, x: Tensor) -> Transform:
        θ = self.net(x)
        s, t = θ.tensor_split(2, dim=1)
        return AffineTransform(s, t)


class ConditionalRQSplineTransformFNN(nn.Module):
    def __init__(
        self,
        lattice_shape: list[int],
        *,
        n_segments: int,
        upper_bound: float,
        **net_kwargs,
    ):
        super().__init__()
        n_lattice = math.prod(lattice_shape)
        n_params = 2 * n_segments - 1
        self.net = make_fnn(
            size_in=n_lattice,
            size_out=n_lattice * (2 * n_segments - 1),
            **net_kwargs,
        )
        self.SplineTransform = partial(
            RQSplineTransform,
            lower_bound=-upper_bound,
            upper_bound=upper_bound,
            bounded=False,
            periodic=False,
        )

    def forward(self, x: Tensor) -> Transform:
        θ = self.net(x)
        w, h, d = θ.tensor_split(3, dim=1)
        return self.SplineTransform(w, h, d)
