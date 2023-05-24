import torch
import torch.nn as nn

from torchlft.typing import Action, BaseAction, Geometry, Tensor

# TODO: Type alias for Tensor | tuple[Tesor, ...] etc etc


class NormalizingFlow(nn.Module):
    def __init__(self, geometry: Geometry, layers):
        super().__init__()
        self.geometry = geometry
        self.layers = nn.ModuleList(layers)

    def forward(
        self, inputs: Tensor | tuple[Tensor, ...]
    ) -> tuple[Tensor, Tensor] | tuple[tuple[Tensor, ...], Tensor]:
        partitions = self.geometry.partition(inputs)

        ldj_total = torch.zeros(len(inputs), device=inputs.device)

        for layer in self.layers:
            partitions, ldj = layer(partitions)
            ldj_total += ldj

            self.on_layer(partitions, ldj_total)

        outputs = self.geometry.restore(partitions)

        return outputs, ldj_total

    def on_layer(self, partitions: tuple[Tensor, ...], ldj: Tensor) -> None:
        pass


class NormalizingFlowComposition(nn.Module):
    def __init__(self, flows: list[NormalizingFlow]):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(
        self, inputs: Tensor | tuple[Tensor, ...]
    ) -> tuple[Tensor, Tensor] | tuple[tuple[Tensor, ...], Tensor]:
        outputs = inputs
        ldj_total = torch.zeros(len(inputs), device=inputs.device)

        for flow in self.flows:
            outputs, ldj = flow(inputs)
            ldj_total += ldj

        return outputs, ldj_total


def compose(*flows: NormalizingFlow) -> NormalizingFlowComposition:
    return NormalizingFlowComposition(*flows)


class BoltzmannGenerator(nn.Module):
    def __init__(
        self,
        base: BaseAction,
        target: Action,
        flow: NormalizingFlow | NormalizingFlowComposition,
    ):
        super().__init__()
        self.base = base
        self.target = target
        self.flow = flow

    def forward(
        self, inputs: Tensor | tuple[Tensor, ...]
    ) -> tuple[Tensor | tuple[Tensor, ...], Tensor]:
        return self.flow(inputs)

    def inverse(
        self, inputs: Tensor | tuple[Tensor, ...]
    ) -> tuple[Tensor | tuple[Tensor, ...], Tensor]:
        return self.flow.inverse(inputs)

    def inference_step(
        self, inputs: Tensor | tuple[Tensor, ...]
    ) -> dict[str, Tensor | tuple[Tensor, ...]]:
        target_action = self.target.compute(inputs)
        outputs, ldj_inverse = self.inverse(inputs)
        base_action = self.base.compute(outputs)
        model_action = base_action - ldj_inverse
        return dict(
            inputs=inputs,
            outputs=outputs,
            base_action=base_action,
            model_action=model_action,
            target_action=target_action,
        )

    def sampling_step(
        self, inputs: Tensor | tuple[Tensor, ...]
    ) -> dict[str, Tensor | tuple[Tensor, ...]]:
        base_action = self.base.compute(inputs)
        outputs, ldj_forward = self(inputs)
        model_action = base_action + ldj_forward
        target_action = self.target.compute(outputs)
        return dict(
            inputs=inputs,
            output=outputs,
            base_action=base_action,
            model_action=model_action,
            target_action=target_action,
        )

    def sample(self, N: int) -> tuple[Tensor | tuple[Tensor, Tensor], Tensor]:
        configs = self.base.sample(N)
        return self.sampling_step(configs)
