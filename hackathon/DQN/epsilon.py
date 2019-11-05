import numpy as np


def epsilon_greedy(q_values, epsilon, num_actions):
    u = np.random.uniform()
    if u <= epsilon:
        a = np.random.randint(low=0, high=num_actions)
    else:
        max_q = np.max(q_values)
        centered_q_values = q_values - max_q
        q_probs = np.exp(centered_q_values) / np.sum(np.exp(centered_q_values))
        a = np.random.choice(a=num_actions, p=q_probs)
    return a

q_values = [1000, 1000, 900]
epsilon = 0.5

a = epsilon_greedy(q_values, epsilon, 3)
print(a)