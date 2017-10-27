from matplotlib import pyplot as plt
from random import gauss
from simple import step


def update_ground_truths(ground_truths):
    ground_truths = [q + gauss(0, 0.01) for q in ground_truths]
    return ground_truths


def train(k, steps, greedyness, step_size=None):
    estimates = [0] * k
    amounts = [0] * k
    ground_truths = [gauss(0, 1)] * k
    actions = []
    rewards = []
    avg_rewards = []
    for s in range(steps):
        ground_truths = update_ground_truths(ground_truths)
        action, reward = step(amounts, estimates, greedyness, ground_truths, k, step_size)
        rewards.append(reward)
        avg_rewards.append(sum(rewards) / len(rewards))
        actions.append(action)
        print('Step', s, 'Reward', reward, 'Q(' + str(action) + ')', estimates[action], 'N(' + str(action) + ')',
              amounts[action])
    optimal_action = ground_truths.index(max(ground_truths))
    optimals = [len(list(filter(lambda a: a == optimal_action, actions[:i + 1]))) / len(actions[:i + 1]) for i in range(len(actions))]
    return avg_rewards, optimals, amounts, estimates


k = 10
steps = 10000
epsilon = .1
alpha = .1
# Sample average
avg_rewards_s, optimals_s, _, _ = train(k, steps, epsilon)
# Constant step size
avg_rewards_c, optimals_c, _, _ = train(k, steps, epsilon, alpha)
plt.xlabel('Steps')
plt.ylabel('Average reward')
plt.plot(avg_rewards_s, label='1/n')
plt.plot(avg_rewards_c, label=str(alpha))
plt.legend(loc='upper right')
plt.savefig('img/2.5.1.png')
plt.show()
plt.xlabel('Steps')
plt.ylabel('Optimal action %')
plt.plot(optimals_s, label='1/n')
plt.plot(optimals_c, label=str(alpha))
plt.legend(loc='upper right')
plt.savefig('img/2.5.2.png')
plt.show()
