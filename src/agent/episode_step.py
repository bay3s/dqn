import torch


class EpisodeStep:

  def __init__(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor):
    """
    Initialize one step in an episode.

    :param state:
    :param action:
    :param reward:
    :param next_state:
    """
    self.state = state
    self.action = action
    self.reward = reward
    self.next_state = next_state
