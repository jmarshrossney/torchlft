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
import pytorch_lightning as pl

from torchlft.phi4.action import parse_couplings, Phi4Action
from torchlft.phi4.base import Phi4BaseAction
from torchlft.phi4.coupling_flow import LayerSpec, make_flow
from torchlft.metrics import metropolis_test, acceptance_rate, effective_sample_size

from torchlft.lightning.optim import OptimizerConfig

from torchlft.typing import Optimizer, Scheduler

from torchlft_examples.phi4.coupling.model import Model


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
        model: Model,
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

    model = Model(
            base=config.base,
            target=target,
            flow=flow,
    )
    config.optim.add_to(model)

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cuda",
        limit_train_batches=config.steps,
        limit_val_batches=5,
        limit_test_batches=1,
        num_sanity_val_steps=0,
        val_check_interval=100,
        logger=True,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        callbacks=[pl.callbacks.ModelSummary(2), pl.callbacks.LearningRateMonitor()],
    )

    trainer.fit(model)
    trainer.validate(model)

    output_path = pathlib.Path(config.output)

    # create output path if it doesn't exist...

    # save checkpoint

    with (output_path / CONFIG_FNAME).open("w") as file:
        file.write(config_yaml)

    return trained_model


if __name__ == "__main__":
    main()
