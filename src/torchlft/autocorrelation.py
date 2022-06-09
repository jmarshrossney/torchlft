import torch
import matplotlib.pyplot as plt


class Autocorrelation:
    def __init__(self, observable: torch.Tensor) -> None:
        if observable.dim() == 1:
            self._observable = observable.unsqueeze(0)
        elif observable.dim() == 2:  # has batch size
            self._observable = observable
        else:
            raise ValueError(
                "Expected input dim to be 1 or 1 but got {observable.dim()}"
            )

    def _compute_autocorrelation(self) -> None:
        autocovariance = torch.nn.functional.conv1d(
            torch.nn.functional.pad(
                self._observable.unsqueeze(1), (0, len(self._observable) - 1)
            ),
            self._observable.unsqueeze(1),
        ).squeeze()
        autocorrelation = autocovariance.div(autocovariance[:, 0])
        self._autocorrelation = autocorrelation

    @property
    def autocorrelation(self) -> torch.Tensor:
        return self._autocorrelation

    @property
    def integrated_autocorrelation(self) -> torch.Tensor:
        pass

    def plot_integrated_autocorrelation(self) -> plt.figure:
        raise NotImplementedError
        # return self._autocorrelation.cumsum()
