from typing import List
import random
from collections import namedtuple
from collections import deque

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))


class ReplayMemory:

  def __init__(self, capacity: int) -> None:
    """
    Initialize an episode.

    :param capacity: The maximum capacity for the replay memory.
    """
    self._transitions = deque([], maxlen=capacity)
    self.capacity = capacity

  def add_step(self, step: Transition) -> Transition:
    """
    Add the results of an episode step to the memory.

    :param step: the episode step to add to the replay memory.

    :return: None | EpisodeStep
    """
    self._transitions.append(step)

    return step

  @property
  def is_full(self) -> bool:
    """
    Returns true if the replay memory is full or over capacity.

    Allowing the memory to run a bit over capacity just to allow episodes to play out.

    :return: bool
    """
    return self.capacity <= len(self._transitions)

  @property
  def transitions(self) -> List[Transition]:
    """
    Return the list of episode steps in the current memory.

    :return: List[EpisodeStep]
    """
    return self._transitions

  def sample(self, num_samples: int) -> list:
    """
    Returns the number of samples requested from the replay memory.

    :param num_samples:

    :return: list
    """
    return random.sample(self._transitions, num_samples)
