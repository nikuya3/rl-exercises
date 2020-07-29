from random import gauss, random, randint

# alpha = 0.4
epsilon = 0.1
k = 10
steps = 1000
ground_truths = [gauss(0, 1) for i in range(k)]
optimal_action = ground_truths.index(max(ground_truths))


def bandit(ground_truths, action):
    return gauss(ground_truths[action], 1)


def update_estimate(old, target, step_size):
    return old + step_size * (target - old)


def step(amounts, estimates, greedyness, ground_truths, k, step_size):
    explore = random() < greedyness
    if explore:
        action = randint(0, k - 1)
    else:
        action = estimates.index(max(estimates))
    reward = bandit(ground_truths, action)
    amounts[action] += 1
    if step_size:
        alpha = step_size
    else:
        alpha = 1 / amounts[action]
    estimates[action] = update_estimate(estimates[action], reward, alpha)
    return action, reward


def train(k, steps, greedyness, ground_truths, step_size=None):
    estimates = [0] * k
    amounts = [0] * k
    rewards = []
    for s in range(steps):
        action, reward = step(amounts, estimates, greedyness, ground_truths, k, step_size)
        rewards.append(reward)
        print('Step', s, 'Reward', reward, 'Q(' + str(action) + ')', estimates[action], 'N(' + str(action) + ')',
              amounts[action])
    return rewards, amounts, estimates


rewards, amounts, estimates = train(k, steps, epsilon, ground_truths)
print('Optimal action ' + str(optimal_action) + ' chosen ' + str(amounts[optimal_action] / sum(amounts)) + ' % of the time')
