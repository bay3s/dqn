**References**
- Fast Gradient-Descent Methods for TD-Learning with Linear Function Approximation (Sutton et al., 2009)
- Playing Atari with Deep Reinforcement Learning (Mnih et al., 2013)
- Deep RL with Double Q-learning (Hasselt et al., 2015)

**Overview**
- DQN is model-free, off-policy, and online.
- Inputs to the policy are observations from the environment and output is a value function estimating future rewards.
- Uses an experience replay mechanism which randomly samples previous transitions to smooth the training distribution over past transitions.
- At each time-step the agent selects an action which is passed to the game emulator.
- The agent only observes an image from the emulator and receives a reward representing the change in game score.
- The goal of the agent is to interact with the emulator with by selecting actions that maximize rewards.
- The optimal action-value Q*(s, a) is defined as the maximum expected return achievable by following any strategy based on the policy $\pi$ after seeing a sequence s and then taking some action a.
$$Q*(s, a) = max_\pi \mathbb{E}[R_t|s+t=s, a_t=a, \pi]$$
- It solves the RL task directly using samples from the emulator without explicitly constructing the state of the environment.

**Related Work**
- Double DQN
- Dueling DQN
- Experience Replay:
  - Prioritized Replay
  - Hindsight Experience Replay

**Challenges**
- Learning form a scalar reward.
- Delay between actions and rewards can be long.
- Deep learning assumes the data samples to be IID, in RL one typically encounters sequences that are highly correlated.
- Data distribution changes as an algorithm learns new behaviours, which can be problematic for deep learning methods that assume a fixed underlying distribution.

**Experience Replay**
- Store the agent's experience at each time-step pooled over many episodes into a replay memory.
- During the inner loop of the algorithm, apply Q-learning updates (aka mini-batch updates) to samples of experience drawn at random from the pool of stored samples.
- Instead of applying Q-learning on state-action pairs as they occur experience replay methods stores these before using them to train the policy.
- Disadvantage is that the random sampling approach does not take into account the importance of transitions based on which ones the agent can learn from the most.
- Advantages:
	- Each step of experience is potentially used in many weight updates which enables data efficiency.
	- Learning directly from consecutive samples is inefficient due to the strong correlations, random sampling breaks these correlations and reduces variance of the updates.
	- When learning on-policy the current parameters determine the next data sample that the parameters are trained on. But, by using experience replay the behavior distribution is averaged over many of its previous states, smoothing out learning and avoiding oscillations or divergence in the parameters.
	- Can replay rare experiences based on a heuristic (eg. information criteria, etc).

**Off-Policy Learning**
- Off-policy learning implies that the learner learns the value of the optimal policy independently of the agent's actions.
- DQN is off-policy because it updates the Q-values using the Q-value of the next state and the greedy action.
- It estimates the return for state-action pairs assuming a greedy policy was followed despite the fact that it's not following a greedy policy.

**Pseudocode**
- initialize replay memory D to capacity N
- initialize state-action-value function Q with random weights
- for episode 1, M do:
	- initialize sequence $s_1 = {x_1}$ and preprocessed sequence $\phi_1 = \phi(s_1)$
	- for t = 1, T do:
		- with probability $\epsilon$ select a random action $a_t$, otherwise select $a_t = max_a Q*(\phi(s_t), a; \theta)$
		- execute action $a_t$ in emulator and observe reward $r_t$ and image $x_{t+1}$
		- set $s_{t+1}  = s_t, a_t, x_{t+1}$ and preprocess $\phi_{t+1} = \phi(s_t + 1)$
		- store transition $(\phi_t, a_t, r_t, \phi_{t+1})$ in D
		- sample random minibatch of transitions $(\phi_j, a_j, r_j, \phi_{j+1})$ from D
		- set
			- $y_j = r_j$ for terminal $\phi_{j+1}$
			- $y_j = r_j + \gamma max_{a'}Q(\phi_{j+1}, a' ; \theta)$ for non-terminal $\phi_{j+1}$
			- perform a gradient descent step on $(y_j - Q(\phi_j, a_j; \theta))^2$
