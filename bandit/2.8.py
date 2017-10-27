# Make a figure analogous to Figure 2.6 for the
# non-stationary case outlined in Exercise 2.5. Include the constant-step-size ε-greedy
# algorithm with α = 0.1. Use runs of 200,000 steps and, as a performance measure for
# each algorithm and parameter setting, use the average reward over the last 100,000
# steps.
from matplotlib import pyplot as plt
from math import exp, log, sqrt
from random import gauss
from simple import bandit, update_estimate, step


def update_ground_truths(ground_truths):
    ground_truths = [q + gauss(0, 0.01) for q in ground_truths]
    return ground_truths


def greedy(k, steps, last_steps, greedyness, step_size=None, optimistic=None):
    if optimistic:
        estimates = [optimistic] * k
    else:
        estimates = [0] * k
    amounts = [0] * k
    ground_truths = [gauss(0, 1)] * k
    rewards = []
    for s in range(1, steps + 1):
        ground_truths = update_ground_truths(ground_truths)
        action, reward = step(amounts, estimates, greedyness, ground_truths, k, step_size)
        if s > steps - last_steps:
            rewards.append(reward)
        print('Step', s, 'Reward', reward, 'Q(' + str(action) + ')', estimates[action], 'N(' + str(action) + ')',
              amounts[action])
    avg_reward = sum(rewards) / len(rewards)
    return avg_reward


def bound(estimate, degree, step, amount):
    if amount <= 0 or step <= 0:
        return 0
    return estimate + degree * sqrt(log(step) / amount)


def ucb(k, steps, last_steps, degree, step_size=None):
    estimates = [0] * k
    amounts = [0] * k
    ground_truths = [gauss(0, 1)] * k
    rewards = []
    for s in range(1, steps + 1):
        ground_truths = update_ground_truths(ground_truths)
        ucbs = [bound(estimates[a], degree, s, amounts[a]) for a in range(k)]
        action = ucbs.index(max(ucbs))
        reward = bandit(ground_truths, action)
        amounts[action] += 1
        if step_size:
            alpha = step_size
        else:
            alpha = 1 / amounts[action]
        estimates[action] = update_estimate(estimates[action], reward, alpha)
        if s > steps - last_steps:
            rewards.append(reward)
        print('Step', s, 'Reward', reward, 'Q(' + str(action) + ')', estimates[action], 'N(' + str(action) + ')',
              amounts[action])
    avg_reward = sum(rewards) / len(rewards)
    return avg_reward


def gradient(k, steps, step_size):
    preferences = [0] * k
    ground_truths = [gauss(0, 1)] * k
    avg_reward = 0
    for s in range(1, steps + 1):
        ground_truths = update_ground_truths(ground_truths)
        # Get probability of choosing an action using softmax
        # if max(preferences) > 700:
        #     print('Too high')
        #     preferences = [p - .5 for p in preferences]
        exponentials = [exp(p) for p in preferences]
        probs = [exponentials[a] / sum(exponentials) for a in range(len(preferences))]
        action = probs.index(max(probs))
        reward = bandit(ground_truths, action)
        # Incremental average update
        avg_reward = avg_reward + 1 / s * (reward - avg_reward)
        # Update preferences with gradients (preference is increased for above-average rewards)
        preferences[action] = preferences[action] + step_size * (reward - avg_reward) * (1 - probs[action])
        for a in range(k):
            if a != action:
                preferences[a] = preferences[a] - step_size * (reward - avg_reward) * probs[a]
    return avg_reward


k = 10
steps = 200000
last_steps = 100000
alpha = .1
plt.xlabel('Hyperparameter')
plt.ylabel('Average reward of last 100000 steps')
# Constant-step-size ε-greedy
epsilons = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4]
avg_rewards = []
for epsilon in epsilons:
    avg_reward = greedy(k, steps, last_steps, epsilon, alpha)
    avg_rewards.append(avg_reward)
plt.plot(epsilons, avg_rewards, label='ε-greedy')
# Constant-step-size ε-greedy with optimistic initialization
epsilon = .1
initial_estimates = [1/16, 1/8, 1/4, 1/2, 1, 2, 4]
avg_rewards = []
for initial_estimate in initial_estimates:
    avg_reward = greedy(k, steps, last_steps, epsilon, alpha, initial_estimate)
    avg_rewards.append(avg_reward)
plt.plot(initial_estimates, avg_rewards, label='ε-greedy with optimistic initialization')
# Upper confidence bound
cs = [1/16, 1/8, 1/4, 1/2, 1, 2, 4]
avg_rewards = []
for c in cs:
    avg_reward = ucb(k, steps, last_steps, c, alpha)
    avg_rewards.append(avg_reward)
plt.plot(cs, avg_rewards, label='UCB')
# Gradient bandit
alphas = [1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 3]
avg_rewards = []
for alpha in alphas:
    avg_reward = gradient(k, steps, alpha)
    avg_rewards.append(avg_reward)
plt.plot(alphas, avg_rewards, label='gradient bandit')
plt.legend(loc='upper right')
plt.savefig('img/2.8.png')
plt.show()
