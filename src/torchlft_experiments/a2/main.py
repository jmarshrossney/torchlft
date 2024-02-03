from jsonargparse import ActionConfigFile, ArgumentParser, Namespace

import torch

from torchlft.nflow.logging import Logger

from torchlft_experiments.a2.model import Model
from torchlft_experiments.a2.train import Trainer

parser = ArgumentParser(prog="train")
parser.add_class_arguments(Model, "model")
parser.add_class_arguments(Trainer, "train")
parser.add_argument("-c", "--config", action=ActionConfigFile)

def main(config: Namespace):

    config = parser.instantiate_classes(config)

    model = config.model
    trainer = config.train
    logger = Logger()

    trainer.train(model, logger)

    def f(z):
        x, ldj = model.flow_forward(z.unsqueeze(0))
        return x.squeeze(0)

    jac = torch.func.jacrev(f)(torch.randn(model.lattice_size))
    
    print(jac)
    print(model.mask * model.weight)
    print(torch.isclose(jac, model.mask * model.weight))
    print(model.cholesky)

    sample, weights = model.weighted_sample(10000)

    cov = torch.cov(sample.transpose(0, 1))

    print(cov - model.covariance)

if __name__ == "__main__":
    config = parser.parse_args()
    main(config)
