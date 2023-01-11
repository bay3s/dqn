"""
Runs the DQN algorithm on the classical control task Cartpole

The action space is a value {0, 1} where 0 = push cart left & 1 = push cart right.

The observation space consists of {position, velocity, angel, angular velocity}
"""
import gym
import torch
import torch.optim as optim
from src.agent.policy import Policy
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import random

from src.agent.replay_memory import ReplayMemory, Transition


MAX_EPISODES = 10_000
MAX_TRAJECTORY = 10
GAMMA = 0.99

EPSILON = 1
EPSILON_ANNEALING = 0.9 / 1_000_000

BATCH_SIZE = 128
TARGET_UPDATE = 25

# variables for optimization
env = gym.make('CartPole-v1')
observation_space = env.observation_space
action_space = env.action_space

# initialize policy network
policy_net = Policy(observation_space.shape[0], action_space.n)

# initialize the target network since DQN learns off-policy
target_net = Policy(observation_space.shape[0], action_space.n)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# optimizer for the policy network
optimizer = optim.RMSprop(policy_net.parameters(), lr=0.01)

# replay memory to keep track of past states & actions.
replay_memory = ReplayMemory(capacity=1_000)
episode_durations = list()

for i_episode in range(1, MAX_EPISODES):
  obs, _ = env.reset()
  state = torch.from_numpy(obs).unsqueeze(0)

  is_done = False
  trajectory_length = 0

  for t in range(MAX_TRAJECTORY):
    if is_done:
      break

    if np.random.random() < EPSILON:
      action = torch.tensor([[random.randrange(env.action_space.n)]], device='cpu', dtype=torch.long)
    else:
      with torch.no_grad():
        action = policy_net(state).argmax().view(1, 1)

    # take action
    new_obs, reward, is_done, _, _ = env.step(action.item())
    next_state = torch.from_numpy(new_obs).unsqueeze(0)
    reward = torch.tensor([reward], device='cpu')

    # update the replay memory
    transition = Transition(state, action, reward, next_state)
    replay_memory.add_step(transition)
    state = next_state
    pass

    # anneal the value of epsilon as we iterate through episodes.
    if EPSILON > 0.1:
      EPSILON -= EPSILON_ANNEALING

    trajectory_length = t
    pass
  # end of for loop

  # optimize the model if we have enough samples in memory
  if replay_memory.is_full:
    sampled_transitions = replay_memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*sampled_transitions))

    # train the neural net
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # run the optimizer
    optimizer.zero_grad()
    loss.backward()

    for param in policy_net.parameters():
      param.grad.data.clamp_(-1, 1)

    optimizer.step()
    pass

  if i_episode % TARGET_UPDATE == 0:
    target_net.load_state_dict(policy_net.state_dict())

  # plot the current trajectories
  episode_durations.append(trajectory_length)
  plt.figure(2)
  plt.clf()
  plt.title('Training...')
  plt.xlabel('Episode')
  plt.ylabel('Duration')
  plt.plot(episode_durations)

  # pause so that the plots can be updated
  plt.pause(0.001)
