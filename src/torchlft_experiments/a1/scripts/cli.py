from jsonargparse import ArgumentParser, Namespace

import torchlft_experiments.a1.scripts.train as train

parser = ArgumentParser(prog="exp1")

subcommands = parser.add_subcommands()
subcommands.add_subcommand("train", train.parser)
# subcommands.add_subcommand("test", test.parser)


def main(config: Namespace) -> None:
    if config.subcommand == "train":
        train.main(config.train)
    elif config.subcommand == "test":
        test.main(config.test)
    elif config.subcommand == "hmc":
        hmc.main(config.hmc)


if __name__ == "__main__":
    config = parser.parse_args()
    main(config)
