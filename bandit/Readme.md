Bandit problems
===
`simple.py`
---
`simple.py` contains a basic k-armed bandit algorithm using ε-greedy action selection and sample-average estimations.
An agent selects one of `k` actions, which results in a reward. The true rewards of the actions - values - are unknown
to the agent. The values are generated according to a unit gaussian distribution. The agent aims to maximize its reward
by updating its estimates of the action values. In each step, the agent takes a exploits its current estimates with a
probability of `1 - ε` to get the highest reward. With a probability of `ε` it explores a new action by sampling a
random action which has suboptimal estimates.
### Hyperparameters:
Parameter | Value | Explanation
--- | ---: | ---
k | 10 | The amount of 'arms', i.e. the amount of possible actions to take
alpha | 0.4 | The step size, which controls the magnitude of estimate updates
epsilon | 0.1 | The probability of exploring non-greedy actions
