# probabilistic-programming
In this repository, we compare two probabilistic programming frameworks: [PyMC3](https://docs.pymc.io) and [Pyro](https://pyro.ai).

## Setup
Create a new conda environment, install [PyTorch](https://pytorch.org) and the remaining requirements:
```
conda create python==3.8 -n probabilistic-programming
conda activate probabilistic-programming
conda install pytorch torchvision -c pytorch
pip install -r requirements.txt
```

## Code
This repository contains two examples — a 1D linear regression and a 1D hierarchical linear regression — that illustrate language features well.

## Advantages of each framework
- PyMC3 has better support for Markov Chain Monte Carlo (MCMC) inference (faster and can parallellize across multiple chains), while Pyro was built for variational inference (although both frameworks support both approaches)
- Pyro make it easier to use neural network components (in a VAE, for example), as it is built on top of PyTorch for automatic differentiation while PyMC3 is built on top of Theano
- PyMC3 + MCMC works best for smaller models, while Pyro can scale to very large models
