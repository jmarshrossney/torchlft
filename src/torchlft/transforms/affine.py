import torch

def affine_forward(x: Tensor, params: Tensor) -> tuple[Tensor, Tensor]:
    s, t = params.split(1, dim=-1)
    return x * s + t, s

def affine_inverse(y: Tensor, params: Tensor) -> tuple[Tensor, Tensor]:
    s, t = params.split(1, dim=-1)
    s_inv = 1 / s
    return (y - t) * s_inv, s_inv

def affine_transform_(

        scale_fn: str = "exponential",
        equivariant: bool = False,
        rescale_only: bool = False,
        ):

    if scale_fn == "exponential":
        scale_fn = lambda s: torch.exp(-s)
    elif scale_fn == "softplus":
        scale_fn = torch.softplus

    if equivariant:
        scale_fn = lambda s: scale_fn(s) + scale_fn(-s)

    if rescale_only:
        n_params = 1
        def handle_params(params: Tensor) -> Tensor:
            return torch.cat([scale_fn(params), torch.zeros_like(params)], dim=-1)
    else:
        n_params = 2
        def handle_params(params: Tensor) -> Tensor:
            s, t = params.split(1, dim=-1)
            return torch.cat([scale_fn(s), t], dim=-1)

    return pointwise_transform_(
            affine_forward,
            affine_inverse,
            handle_params,
            n_params,
    )



    class AffineTransform:
        def __init__(self, params: Tensor):
            self.params = handle_params(params)

        def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
            return affine_forward(x, self.params)

        def inverse(self, y: Tensor) -> tuple[Tensor, Tensor]:
            return affine_inverse(y, self.params)


    return AffineTransform
