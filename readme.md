**Intro**
- First deep learning model to successfully learn control policies from high-dimensional sensory input using RL.
- Uses a CNN trained with a variant of Q-laerning
- Inputs to the CNN are raw pixles and output is a value function estimating future rewards.
- Uses an experience replay mechanism which randomly samples previous transitions, and thereby smooths the training distribution over many past behaviors.

**Overview**
- At each time-step the agent selects an action which is passed to the game emulator.
- The agent only observes an image from the emulator and receives a reward representing the change in game score.
- The goal of the agent is to interact with the emulator with by selecting actions that maximize rewards.
- Optimal action-value Q*(s, a) is defined as the maximum expected return achievable by following any strategy based on the policy $\pi$ after seeing a sequence s and then taking some action a.
$$Q*(s, a) = max_\pi \mathbb{E}[R_t|s+t=s, a_t=a, \pi]$$
- DQN is model-free, it solves the RL task directly using samples from the emulator without explicitly constructing the state of the environment.

**Related Work**
- TD-Gammon
- Experience Replay

**Challenges**
- Learning from a scalar reward.
- Delay between actions and rewards can be long.
- Deep learning algos assume the data samples to be IID, in RL one typically encounters sequences that are highly correlated.
- The data dsistribution changes as an algorithm learns new behaviours, which can be problematic for deep learning methods that asusme a fixed underlying distribution.

**Experience Replay**
- Store the agent's experience at each time-step pooled over many episodes into a replay memory.
- During the inner loop of the algorithm, apply Q-learning updates (aka mini-batch updates) to samples of experience drawn at random from the pool of stored samples.
- Instead of applying Q-learning on state-action pairs as they occur experience replay methods stores these before using them to train the policy.
- Advantages:
	- Each step of experience is potentially used in many weight updates which enables data efficiency.
	- Learning directly from consecutive samples is inefficient due to the strong correlations, random sampling breaks these correlations and reduces variance of the updates.
	- When learning on-policy the current parameters determine the next data sample that the parameters are trained on. But, by using experience replay the behavior distribution is averaged over many of its previous states, smoothing out learning and avoiding oscillations or divergence in the parameters.
	- Can replay rare experiences based on a heuristic (eg. informatino criteria, etc).
- Disadvantage is that the random sampling approach does not take into account the importance of transitions based on which ones the agent can learn from the most.

**Off-Policy Learning**
- While using DQN it's necessary to learn off-policy.
- Off-policy learning implies that the learner learns the value of the optimal policy independently of the agent's actions.
- DQN is off-policy because it updates the Q-values using the Q-value of the next state and the greedy action.
- It estimates the return for stat-action pairs assuming a greedy policy was followed despite the fact that it's not following a greedy policy.

**Pseudocode**
- Initialize replay memory D to capacity N
- Initialize state-action-value function Q with random weights
- For episode 1, M do:
	- Initialize sequence $s_1 = {x_1}$ and preprocessed sequence $\phi_1 = \phi(s_1)$
	- for t = 1, T do:
		- With probability $\epsilon$ select a random action $a_t$, otherwise select $a_t = max_a Q*(\phi(s_t), a; \theta)$
		- Execute action $a_t$ in emulator and observe reward $r_t$ and image $x_{t+1}$
		- Set $s_{t+1}  = s_t, a_t, x_{t+1}$ and preprocess $\phi_{t+1} = \phi(s_t + 1)$
		- Store transition $(\phi_t, a_t, r_t, \phi_{t+1})$ in D
		- Sample random minibatch of transitions $(\phi_j, a_j, r_j, \phi_{j+1})$ from D
		- Set
			- $y_j = r_j$ for terminal $\phi_{j+1}$
			- $y_j = r_j + \gamma max_{a'}Q(\phi_{j+1}, a' ; \theta)$ for non-terminal $\phi_{j+1}$
			- Perform a gradient descent step on $(y_j - Q(\phi_j, a_j; \theta))^2$


