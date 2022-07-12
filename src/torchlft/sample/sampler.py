import os
import pathlib
from typing import Optional, Union

from jsonargparse.typing import PositiveInt
import torch
import torch.utils.tensorboard as tensorboard
import tqdm

from torchlft.sample.algorithms import SamplingAlgorithm


class Sampler:
    def __init__(
        self,
        algorithm: SamplingAlgorithm,
        output_dir: Optional[Union[str, os.PathLike]] = None,
    ):
        self._algorithm = algorithm
        if output_dir is not None:
            self._output_dir = pathlib.Path(str(output_dir)).resolve()
        else:
            self._output_dir = None
        self._run_idx = 0

        if hasattr(self._algorithm, "sweep_length"):
            self._update = self._sweep
        else:
            self._update = self._step

        self.reset()

    @property
    def algorithm(self) -> SamplingAlgorithm:
        """
        Pointer to the sampling algorithm.
        """
        return self._algorithm

    @property
    def output_dir(self) -> pathlib.Path:
        """
        Directory for sampling outputs.
        """
        return self._output_dir

    @property
    def run_idx(self) -> PositiveInt:
        return self._run_idx

    def _step(self) -> None:
        self._algorithm()
        self._algorithm.on_step()

    def _sweep(self) -> None:
        for _ in range(self._algorithm.sweep_length):
            self._step()
        self._algorithm.on_sweep()

    def _sample(self, interval: PositiveInt) -> None:
        for _ in range(interval):
            self._update()
        self._algorithm.on_sample()

    def reset(self) -> None:
        self._run_idx += 1
        if self._output_dir is not None:
            log_dir = str(self._output_dir / "logs" / f"run_{self._run_idx}")
            self._logger = tensorboard.writer.SummaryWriter(log_dir)
            self._algorithm.logger = self._logger
        self._algorithm.init()

    def thermalise(self, steps_or_sweeps: PositiveInt) -> None:
        with tqdm.trange(
            steps_or_sweeps,
            desc="Thermalising",
            postfix=self._algorithm.pbar_stats,
        ) as pbar:
            for _ in pbar:
                self._update()
                pbar.set_postfix(self._algorithm.pbar_stats)

    def sample(
        self,
        size: PositiveInt = 1,
        interval: PositiveInt = 1,
    ):
        configs = torch.full_like(self._algorithm.state, float("NaN")).repeat(
            size, *(1 for _ in self._algorithm.state.shape)
        )
        with tqdm.trange(
            size, desc="Sampling", postfix=self._algorithm.pbar_stats
        ) as pbar:
            for i in pbar:
                self._sample(interval)
                configs[i] = self._algorithm.state
                pbar.set_postfix(self._algorithm.pbar_stats)

        self._algorithm.on_final_sample()

        return configs
