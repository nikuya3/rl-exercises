# Design and conduct an experiment to demonstrate
# the difficulties that sample-average methods have for nonstationary problems. Use a
# modified version of the 10-armed testbed in which all the q ∗ (a) start out equal and
# then take independent random walks (say by adding a normally distributed increment
# with mean zero and standard deviation 0.01 to all the q ∗ (a) on each step). Prepare
# plots like Figure 2.2 for an action-value method using sample averages, incrementally
# computed, and another action-value method using a constant step-size parameter,
# α = 0.1. Use ε = 0.1 and longer runs, say of 10,000 steps.

from random import gauss, random, randint

alpha = 0.4
epsilon = 0.1
steps = 1000
k = 10
estimates = [0] * k
amounts = [0] * k
ground_truths = [gauss(0, 1) for i in range(k)]
optimal_action = ground_truths.index(max(ground_truths))


def bandit(action):
    return gauss(ground_truths[action], 1)


def update_estimate(old, target, step_size):
    return old + step_size * (target - old)


for step in range(1000):
    explore = random() < epsilon
    if explore:
        action = randint(0, k - 1)
    else:
        action = estimates.index(max(estimates))
    reward = bandit(action)
    amounts[action] += 1
    estimates[action] = update_estimate(estimates[action], reward, alpha)
    print('Step', step, 'Reward', reward, 'Q(' + str(action) + ')', estimates[action], 'N(' + str(action) + ')', amounts[action])
print('Optimal action ' + str(optimal_action) + ' chosen ' + str(amounts[optimal_action] / sum(amounts)) + ' % of the time')
