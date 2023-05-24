import pathlib
from typing import Optional, Any

from jsonargparse import (
    ArgumentParser,
    ActionConfigFile,
    Namespace,
    class_from_function,
)
from jsonargparse.typing import PositiveInt, Path_dc
from tqdm import trange

from torchlft.phi4.action import parse_couplings, Phi4Action
from torchlft.phi4.base import Phi4BaseAction
from torchlft.phi4.coupling_flow import LayerSpec, make_flow
from torchlft.nflow import BoltzmannGenerator
from torchlft.utils.optim import OptimizerConfig
from torchlft.metrics import metropolis_test, acceptance_rate, effective_sample_size

from torchlft.typing import Optimizer, Scheduler

parser = ArgumentParser(parser_mode="omegaconf")
parser.add_argument("--lattice_shape", type=tuple[PositiveInt, PositiveInt])
parser.add_argument("--base", type=Phi4BaseAction)
parser.add_argument("--target", type=dict[str, float])
# parser.add_dataclass_arguments(LayerSpec, "layers.help", required=False)
parser.add_argument("--layers", type=list[LayerSpec])
#parser.add_class_arguments(OptimizerConfig, "optim", as_group=False)
parser.add_argument("--optim", type=OptimizerConfig)
parser.add_argument("--steps", type=PositiveInt)
parser.add_argument("--batch_size", type=PositiveInt)

parser.add_argument("-o", "--output", type=Path_dc)
parser.add_argument("-c", "--config", action=ActionConfigFile)

CONFIG_FNAME = "config.yaml"


def train(
        model: BoltzmannGenerator,
        batch_size: int,
        steps: int,
        optimizer: OptimizerConfig,
        ):

    optimizer, scheduler = optimizer.init(model)

    
    with trange(steps, desc="loss:") as pbar:
        for _ in pbar:
            outputs = model.sample(batch_size)
            log_weights = outputs["model_action"] - outputs["target_action"]
            loss = log_weights.mean().negative()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            pbar.set_description_str(f"{float(loss):.2g}")

    return model

def main(config: Optional[dict[str, Any]] = None) -> None:
    config = (
        parser.parse_object(config)
        if config is not None
        else parser.parse_args()
    )

    config_yaml = parser.dump(config, skip_none=False)

    config = parser.instantiate_classes(config)

    print(config)

    target = Phi4Action(**config.target)

    flow = make_flow(config.lattice_shape, config.layers)

    model = BoltzmannGenerator(
            base=config.base,
            target=target,
            flow=flow,
    )

    trained_model = train(
            model,
            config.batch_size,
            config.steps,
            config.optim,
    )

    outputs = trained_model.sample(10000)
    log_weights = outputs["model_action"] - outputs["target_action"]
    idx, h = metropolis_test(log_weights)
    acc = acceptance_rate(h)
    ess = effective_sample_size(log_weights)
    print(f"""
Acceptance: {acc}
Effective Sample Size: {ess}
""")

    output_path = pathlib.Path(config.output)

    # create output path if it doesn't exist...

    # save checkpoint

    with (output_path / CONFIG_FNAME).open("w") as file:
        file.write(config_yaml)

    return trained_model


if __name__ == "__main__":
    main()
