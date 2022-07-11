import os
import pathlib
from typing import Optional, Union

from jsonargparse.typing import PositiveInt
import torch
import tqdm

from torchlft.sample.algorithms import SamplingAlgorithm


class Sampler:
    def __init__(self, output_dir: Optional[Union[str, os.PathLike]] = None):
        self._output_dir = pathlib.Path(str(output_dir)).resolve()

    def _step(self, model: SamplingAlgorithm) -> None:
        model()
        # model.on_step_end()

    def _sweep(self, model: SamplingAlgorithm) -> None:
        for _ in range(model.sweep_length):
            self._step(model)
        # model.on_sweep_end

    def _sample(self, model: SamplingAlgorithm, interval: PositiveInt) -> None:
        if hasattr(model, "sweep_length"):
            update = self._sweep
        else:
            update = self._step

        for _ in range(interval):
            update(model)
        # model.on_sample()

    def thermalise(
        self, model: SamplingAlgorithm, steps_or_sweeps: PositiveInt
    ) -> None:
        if hasattr(model, "sweep_length"):
            update = self._sweep
        else:
            update = self._step

        with tqdm.trange(
            steps_or_sweeps, desc="Thermalising", postfix=model.pbar_stats
        ) as pbar:
            for _ in pbar:
                update(model)
                pbar.set_postfix(model.pbar_stats)

    def sample(
        self,
        model: SamplingAlgorithm,
        size: PositiveInt = 1,
        interval: PositiveInt = 1,
    ):
        configs = torch.full_like(model.state, float("NaN")).repeat(
            size, *(1 for _ in model.state.shape)
        )
        with tqdm.trange(
            size, desc="Sampling", postfix=model.pbar_stats
        ) as pbar:
            for i in pbar:
                self._sample(model, interval)
                configs[i] = model.state
                pbar.set_postfix(model.pbar_stats)

        return configs

    def autocorrelation(self):
        pass
