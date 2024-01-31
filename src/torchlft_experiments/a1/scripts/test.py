from jsonargparse import ActionConfigFile, ArgumentParser, Namespace
from jsonargparse.typing import Path_drw, PositiveInt
import torch

from torchlft.nflow.io import TrainingDirectory, load_model_from_checkpoint

from torchlft_experiments.a1.scripts.train import parser as train_parser

from torchlft.scalar.observables import two_point_correlator

parser = ArgumentParser(prog="test")
#parser.add_class_arguments()

parser.add_argument("model", type=Path_drw)
parser.add_argument("--step", type=PositiveInt)
parser.add_argument("-c", "--config", action=ActionConfigFile)

parser.add_argument("--batch_size", type=PositiveInt, default=10000)
parser.add_argument("--n_batches", type=PositiveInt, default=10)

def main(config: Namespace) -> None:
    print(config)

    train_dir = TrainingDirectory(config.model)

    model = load_model_from_checkpoint(train_dir, train_parser, step=config.step)

    sample, pweights = model.weighted_sample(config.batch_size, config.n_batches)
    print(sample.shape, pweights.shape)
    
    sample, indices = model.metropolis_sample(config.batch_size, config.n_batches)
    print(sample.shape, indices.shape)

    print(sample[indices].shape)

    indices = indices.tolist()

    print(len(set(indices)) / len(indices))


    from itertools import pairwise
    n_acc = indices[0] + sum([j - i for (i, j) in pairwise(indices)])
    print(n_acc / len(indices))


    sample = sample[indices]
    corr = two_point_correlator(sample)

    print(corr)

    import matplotlib.pyplot as plt

    plt.imshow(corr)
    plt.show()
