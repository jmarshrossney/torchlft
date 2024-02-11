from jsonargparse import ArgumentParser, Namespace

import torchlft.scripts.train as train
#import torchlft_experiments.a1.scripts.test as test

parser = ArgumentParser(prog="lft")

subcommands = parser.add_subcommands()
subcommands.add_subcommand("train", train.parser)
#subcommands.add_subcommand("test", test.parser)


def main(config: Namespace) -> None:
    if config.subcommand == "train":
        train.main(config.train)
    elif config.subcommand == "test":
        test.main(config.test)


if __name__ == "__main__":
    config = parser.parse_args()
    main(config)
