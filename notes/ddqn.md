**References**
- Deep Reinforcement Learning with Double Q-Learning (Hasselt et al., 2015)
- Playing Atari with Deep Reinforcement Learning (Mnih et al., 2013)

**Abstract**
- Q-learning is known to overestimate action values under certain conditions.
- Shows that DQN suffers from substantial overestimations in some cases.
- Generalizes the idea behind the tabular double q-learning algorithm to work with large-scale function approximation.
- Proposes a specific adaptation to DQN and shows that the resulting algorithm reduces overestimations and leads to much better performance in several games.

**Intro**
- RL aims to learn good policies for sequential decision-making problems by optimizing a cumulative reward signal.
- Q-learning seems to learn unrealistically high action value because it includes a maximization step over estimated action values that gives preference to overestimations.
- Previous work attributes overestimation to noise, flexible function approximation.
- Given that imprecise action-value functions are the norm during learning, overestimates are much more common than previously imagined.
- Overoptimistic estimates may not be a problem in and of themselves if all values were uniformly higher which preserves the relative values and preference given to each action.
	- Optimism in the face of uncertainty is a well-known exploration technique.
- But, if overestimations are not uniform and not concentrated at states where we would like to explore - then they negatively affect the quality of the resulting policy.

**DQN**
- The target used by DQN is:

$$
Y_t^{\mathrm{DQN}} \equiv R_{t+1}+\gamma \max Q\left(S_{t+1}, a ; \boldsymbol{\theta}_t^{-}\right)
$$

**DDQN**
- The max operator in DQN uses the same values both to select and to evaluate an action.
- Makes it more likely to select overestimated values, resulting in overoptimistic value estimates.
- Solution is to decouple the selection and evaluation.
- In tabular double q-learning, this is done by learning two value functions by assigning each experience randomly to update one of the two value functions such that there are two sets of weights.
- For each update one set of weights is used to determine the greedy policy and the other to determine its value - untangling the selection and evaluation in q-learning gives:
$$
Y_t^{\mathrm{Q}}=R_{t+1}+\gamma Q\left(S_{t+1}, \underset{a}{\operatorname{argmax}} Q\left(S_{t+1}, a ; \boldsymbol{\theta}_t\right) ; \boldsymbol{\theta}_t\right)
$$

**Pseudocode Comparison**

*DQN Bellman Updates*

> state = experience_replay.state
> next_state = experience_replay.next_state
> 
> state_max_q = argmax(online_network.predict(state))
> next_state_max_q = argmax(target_network.predict(next_state))
> 
> expected_q = reward + discount_factor * next_state_max_q
> loss = LossFunction(predicted_q, expected_q)

*DDQN Bellman Updates* 

> state = experience_replay.state
> next_state = experience_replay.next_state
> 
> state_max_q = argmax(online_network.predict(state))
> next_state_q_values = target_network.predict(next_state)
> 
> next_state_action = argmax(online_network.predict(next_state))
> next_state_action_value = next_state_q_values.where(action = next_action)
> 
> expected_q = reward + discount_factor * next_state_action_value
> loss = LossFunction(predicted_q, expected_q)

Noteworthy in the DDQN Bellman update is that:
- The selection of the action is due to the online weights.
- Similar to DQN we are still estimating the value of the greedy policy according to the current values as defined by $\theta_t$.
- But, we use the target policy weights to to fairly evaluate the value of this policy.
- Target policy weights can also be updated symmetrically by switching roles with the online policy.

**Lowerbound for Overestimates**
- In tabular q-learning and DQN, if the aciton values contain random uniformly distributed errors then each target is overestimated which aysmptotically lead to sub-optimal policies.
- Estimation errors of any kind can induce an upward bias regardless of source (eg. environmental noise, function approximation, etc.).
- Thrun & Schwartz give an upper bound to the overestimation for a specific setup if the random errors are uniformly distributed in an interval $[- \epsilon, \epsilon ]$:
$$\gamma \epsilon \frac{m - 1}{m + 1}$$
- Consider a state in which all optimal action values are equal at $Q*(s, a) = V*(s)$ for some $V*(s)$.
- Let $Q_t$ be arbitrary value estimates that are on the whole unbiased in the sense that 
$$\sum_a (Q_t(s, a) - V*(s)) = 0$$
- But, that are not all correct so that:
$$\frac{1}{m}\sum_a (Q_t(s, a) - V*(s))^2 = C \text{ where } C > 0$$
- Under these conditions we get a tight lower bound such that:
$$\max_a Q_t(s, a) \geq V*(s) + \sqrt{\frac{C}{m - 1}}$$
- Full proof can be found in Appendix 1 in Hasselt et al., 2015.
- The lower bound in the theorem decreases with the number of actions, but this is an artifact of considering the lower bound which requires specific values to be attained - it is more typical for the overestimations of DQN to increase with the number of actions while DDQN is unbiased.

**Robustness to Human Starts**
- In deterministic games, a unique starting point could lead the learner to remember sequences.
- By testing the agents from various starting points we can test whether the found solutions generalize well, and that the environments provide a challenging testbed for agents.

**Results**
- Overestimations of Q-values by itself is not a problem, but overestimation combined with bootstrapping (estimating something based on another estimate - in the case of DQN estimating the current action value Q using an estimate of the future Q).
- Overestimation combined with bootstrapiing has the negative effect of:
	- Propagating the wrong relative information about which states are more valuable than others
	- Affects the quality of the learnt policies.
- DDQN has a similar update as DQN excep we replay the target with the following equation:

$$Y_t^{DoubleDQN} = R_{t+1} + \gamma Q(S_{t+1}, arg\max_a Q(S_{t+1}, a; \theta_t); \theta_t^-)$$
- THe learning curves for DDQN in the experiments are much closer tot hte true value of the final policy, and DDQN seems to not just produce better value estimates but also better policies.
- We might be tempted to attribute DQN's instability to its off-policy learning with function approximation but this is not the case since DDQN is stable - so we can attribute DQN's instability atleast partially to its overoptimism.
- DDQN is more robust to challenging evaluations suggesting good generaliztion and that soutions do not exploit determinism.

**Remarks**
- DQN is overoptimistic in large-scale problems even if these are deterministic due to the inherent estimation errors of learning - these are more common than previously acknowledged.
- DDQN can be used to reduce overoptimism leading to more stable policies and better learning.
- DDQN does not require additional netowrks or parameters in comparison to DQN.
- DDQN finds better policies than DQN.
