from jsonargparse import ArgumentParser

import torchlft_experiments.a1.scripts.cli as a1

parser = ArgumentParser(prog="exp")

subcommands = parser.add_subcommands()
subcommands.add_subcommand("a1", a1.parser)

def main(config):
    if config.subcommand == "a1":
        a1.main(config.a1)

if __name__ == "__main__":
    config = parser.parse_args()
    main(config)
