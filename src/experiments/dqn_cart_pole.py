from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import gym

import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from src.agents import DQN
from src.policies import DQNCartPolePolicy
from src.replays import VanillaReplay


cart_pole_env = gym.make('CartPole-v1')
policy = DQNCartPolePolicy(cart_pole_env.observation_space.shape[0], cart_pole_env.action_space.n)
replay_memory = VanillaReplay(capacity = 500)
optimizer = Adam(policy.parameters(), lr = 0.001)

dqn = DQN(
  env = cart_pole_env,
  policy = policy,
  replay_memory = replay_memory,
  replay_size = 32,
  optimizer = optimizer,
  discount_rate = 0.999,
  max_epsilon = 1.,
  min_epsilon = 0.1,
  epsilon_decay = 1e-3,
  target_update_steps = 50
)

writer = SummaryWriter()
start_state, _ = cart_pole_env.reset()
start_state = torch.from_numpy(start_state.astype(np.float32))
writer.add_graph(dqn.policy, input_to_model = start_state, verbose = False)

max_episodes = 500
mean_rewards = list()
rewards_last_10 = list()
epsilon_values = list()

plt_epsilon = list()
plt_rewards = list()

for epi in tqdm(range(max_episodes)):
  episode_transitions = dqn.play_episode(tune = True)
  rewards_last_10.append(np.sum(list(zip(*episode_transitions))[2]))

  if epi % 10 == 0:
    writer.add_scalar('dqn/mean_reward', np.mean(rewards_last_10), epi)
    writer.add_scalar('dqn/epsilon', dqn.epsilon, epi)

    plt_rewards.append(np.mean(rewards_last_10))
    plt_epsilon.append(dqn.epsilon)

    rewards_last_10.clear()
    pass

plt.plot(plt_rewards, label = 'Mean Reward / 10 Episodes')
plt.plot(plt_epsilon, label = 'Epsilon')
plt.legend()
plt.show()
plt.savefig('')

print('Training complete.')
pass
