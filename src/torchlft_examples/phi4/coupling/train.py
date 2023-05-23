import pathlib
from typing import Optional, Any

from jsonargparse import (
    ArgumentParser,
    ActionConfigFile,
    Namespace,
    class_from_function,
)
from jsonargparse.typing import PositiveInt, Path_dc

from torchlft.phi4.action import parse_couplings, Phi4Action
from torchlft.phi4.base import Phi4BaseAction
from torchlft.phi4.coupling_flow import LayerSpec, make_flow
from torchlft.utils.optim import OptimizerConfig

parser = ArgumentParser(parser_mode="omegaconf")
parser.add_argument("--lattice_shape", type=tuple[PositiveInt, PositiveInt])
# parser.add_argument("--action", type=dict[str, float])
parser.add_argument("--base", type=Phi4BaseAction)
# parser.add_method_arguments(Phi4Couplings, "ising", "couplings")
# parser.add_argument("--action", type=Phi4Action)
# parser.add_class_arguments(Phi4Action, "action")
# parser.add_dataclass_arguments(LayerSpec, "layers.help", required=False)
# parser.add_argument("--layers", type=list[LayerSpec])
# parser.add_dataclass_arguments(OptimizerConfig, None, as_group=False)
# parser.add_argument("--steps", type=PositiveInt)

parser.add_argument("-o", "--output", type=Path_dc)
parser.add_argument("-c", "--config", action=ActionConfigFile)

CONFIG_FNAME = "config.yaml"


def train():
    ...


def main(config: Optional[dict[str, Any]] = None) -> None:
    config = (
        parser.parse_object(config)
        if config is not None
        else parser.parse_args()
    )

    config_yaml = parser.dump(config, skip_none=False)

    config = parser.instantiate_classes(config)

    print(config)

    action = Phi4Action(**config.couplings)

    base = DiagonalGaussianAction(config.lattice_shape)

    trained_model = train()

    output_path = pathlib.Path(config.output)
    print(output_path.exists())

    # save checkpoint

    with (output_path / CONFIG_FNAME).open("w") as file:
        file.write(config_yaml)

    return trained_model


if __name__ == "__main__":
    main()
