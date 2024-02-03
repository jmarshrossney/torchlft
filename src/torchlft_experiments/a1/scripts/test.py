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
parser.add_argument("--n_batches", type=PositiveInt, default=100)

from torchlft.scalar.actions import GaussianActionLA, FreeScalarAction

def test_actions_agree(φ):

    _, L, T, _ = φ.shape
    m_sq = 1.5

    S1 = FreeScalarAction(m_sq)(φ)

    φ = φ.flatten(1)
    S2 = GaussianActionLA(L, m_sq, 2)(φ)

    assert S1.shape == S2.shape
    assert torch.allclose(S1, S2, atol=1e-6)

def main(config: Namespace) -> None:
    #print(config)

    train_dir = TrainingDirectory(config.model)

    model = load_model_from_checkpoint(train_dir, train_parser, step=config.step)

    sample, pweights = model.weighted_sample(config.batch_size, config.n_batches)
    #print(sample.shape, pweights.shape)
    
    sample, indices = model.metropolis_sample(config.batch_size, config.n_batches)
    #print(sample.shape, indices.shape)

    indices = indices.tolist()
    print("Acceptance: ", len(set(indices)) / len(indices))


    φ = sample[indices]

    test_actions_agree(φ)

    cov = torch.cov(φ.flatten(1).transpose(0, 1))
    #corr = two_point_correlator(sample)
    print("var = \n", φ.var())

    print("cov = \n", cov)
    
    L = sample.shape[1]
    m_sq = 1.
    S_gauss = GaussianActionLA(L, m_sq, 2)
    Σ = S_gauss.covariance
    print("Σ = \n", Σ)

    import matplotlib.pyplot as plt

    plt.imshow(cov)
    
    #plt.imshow((corr - Σ) / Σ.abs())
    
    plt.colorbar()
    plt.show()
