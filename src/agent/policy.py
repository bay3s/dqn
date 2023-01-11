import torch
import torch.nn as nn


class Policy(nn.Module):

  def __init__(self, input_size: int, output_size: int) -> None:
    """
    Constructor for the policy neural net.

    The neural network is configured based on the original DQN paper (Silver et al. 2013).

    :param input_size: The size of the state that is being observed by the agent.
    :param output_size: Should correspond to the number of actions that the agent takes.
    """
    super().__init__()

    layers = [
      nn.Linear(input_size, 512),
      nn.ReLU(),
      nn.Linear(512, 1024),
      nn.ReLU(),
      nn.Linear(1024, 1024),
      nn.ReLU(),
      nn.Linear(1024, 512),
      nn.ReLU(),
      nn.Linear(512, output_size),
    ]

    self.model = nn.Sequential(*layers)

  def forward(self, X: torch.Tensor) -> torch.Tensor:
    return self.model(X)
