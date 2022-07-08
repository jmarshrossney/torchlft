from collections.abc import Mappable
import logging
import os
import pathlib
from typing import Any

import torch
from jsonargparse.typing import PositiveInt
import tqdm

log = logging.getLogger(__name__)


class Sampler(torch.nn.Module):
    def __init__(
        self,
        output_dir: Union[str, os.PathLike] = ".",
        thermalisation: PositiveInt = 1,
        sample_interval: PositiveInt = 1,
        **hparams,
    ):
        super().__init__()

        self.output_dir = pathlib.Path(str(output_dir)).resolve()
        self.thermalisation = thermalisation
        self.sample_interval = sample_interval
        self.hparams = hparams

        self.rng = torch.Generator(device="cpu")

        self.runs = []

        self.register_forward_pre_hook(self.forward_pre_hook)
        self.register_forward_hook(self.forward_post_hook)

    @property
    def config_file(self) -> pathlib.Path:
        return self.output_dir / "sampler_config.pt"

    @property
    def output_file(self) -> pathlib.Path:
        return self.output_dir / "sample_{self.run_idx}.pt"

    def save(self) -> None:
        torch.save(self.state_dict(), self.config_file)

    def load(self) -> None:
        try:
            state_dict = torch.load(self.config_file)
        except Exception as exc:
            raise Exception("Unable to load sampler") from exc
        else:
            self.load_state_dict(state_dict)

    def set_extra_state(self, state: dict) -> None:
        assert isinstance(state, dict), f"expected dict, but got {type(state)}"
        self.thermalisation = state["thermalisation"]
        self.sample_interval = state["sample_interval"]
        self.hparams = state["hparams"]
        self.runs = state["runs"] + self.runs  # append recent runs

    def get_extra_state(self) -> dict:
        return {
            "thermalisation": self.thermalisation,
            "sample_interval": self.sample_interval,
            "hparams": self.hparams,
            "runs": self.runs,
        }

    @staticmethod
    def metropolis_test(delta_log_weight: torch.Tensor) -> bool:
        return delta_log_weight > 0 or delta_log_weight.exp() > torch.rand(
            1, generator=self.rng
        )

    def init(self, init_state):
        self.curr_state = init_state
        # save rng state
        
        self.thermalising = True
        pbar = tqdm.trange(self.thermalisation, desc="Thermalising")
        for _ in pbar:
            self.forward()

        self.thermalising = False


    def sample(
        self, size: PositiveInt, interval: PositiveInt
    ) -> torch.Tensor:
        
        self.sample = {k: [] for k in self.curr_state.keys()}

        pbar = tqdm.trange(sample_size, desc="Sampling")
        for _ in pbar:
            for __ in range(sample_interval):
                self.global_step += 1
                _ = self()

    @staticmethod
    def forward_pre_hook(self_, input) -> None
        self_.global_step += 1


    def forward(self) -> Any:
        raise NotImplementedError


    @staticmethod
    def forward_post_hook(self_, input, output) -> None:
        if self_.thermalising:
            return

        if self_.global_step % self_.sample_interval == 0:
            if isinstance(output, torch.Tensor):
                self.sample.append(output)
            elif isinstance(output, Iterable):
                for i, v in enumerate(output):
                    self.sample[i].append(v)
            elif isinstance(output, Mappable):
                for k, v in self.output.items():
                    self.sample[k].append(v)
            else:
                raise TypeError("Expected forward to return Tensor, Iterable or Mappable")


class OneShotSampler(Sampler):
    def __init__(
        self,
        generator: Iterator[torch.Tensor, torch.Tensor],
        output_dir: Union[str, os.PathLike] = ".",
    ):
        super().__init__(output_dir)
        self.generator = generator

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        proposal, log_weight = next(self.generator)

        delta_log_weight = log_weight - self.curr_state["log_weight"]

        if self.metropolis_test(delta_log_weight):
            self.curr_state.update(config=proposal, log_weight=log_weight)


class FlowBasedSampler(OneShotSampler):
    def __init__(
        self,
        model: torchnf.models.BoltzmannGenerator,
        output_dir: Union[str, os.PathLike] = ".",
    ):
        super().__init__(model.generator(), output_dir)
        self.model = model
