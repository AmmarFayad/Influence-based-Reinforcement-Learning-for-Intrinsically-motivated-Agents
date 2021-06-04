# Influence-based Reinforcement Learning for Intrinsically-motivated Agents
## Overview
This repo provides an implementation of the algorithm 1 (pi-mu framework, see paper for further details) in Pytorch

## Dependencies
* Python 3.4
* PyTorch 0.1.9
* OpenAI Gym
* Numpy 1.14.5
* Multi-Agent Particle Environment

## Code structure

- `model.py`: contains specifications for actor and critic networks.
- `autoencoder.py`: contains specifications for autoencoder network.
- `update_autoencoders.py`: contains update method for autoencoder.
- `trainer.py`: contains update methods for the parameters of pi's and mu's critics and policy networks.
- `main.py`: evaluation and run of the algorithm.
