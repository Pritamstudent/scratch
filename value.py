class SimpleMDP:
    def __init__(self, size=4):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4
        self.start_state = 0
        self.goal_state = size * size - 1
        self.state = self.start_state
        self.P = self.create_transition_probabilities()
    
    def create_transition_probabilities(self):
        P = {}
        for state in range(self.n_states):
            P[state] = {a: [] for a in range(self.n_actions)}
            for action in range(self.n_actions):
                next_state, reward, done = self.get_next_state(state, action)
                P[state][action].append((1.0, next_state, reward, done))
        return P
    
    def get_next_state(self, state, action):
        row, col = divmod(state, self.size)
        if action == 0:  # Up
            row = max(0, row - 1)
        elif action == 1:  # Down
            row = min(self.size - 1, row + 1)
        elif action == 2:  # Left
            col = max(0, col - 1)
        elif action == 3:  # Right
            col = min(self.size - 1, col + 1)
        next_state = row * self.size + col
        reward = 0
        done = False
        if next_state == self.goal_state:
            reward = 1
            done = True
        return next_state, reward, done

    def reset(self):
        self.state = self.start_state
        return self.state

    def step(self, action):
        next_state, reward, done = self.get_next_state(self.state, action)
        self.state = next_state
        return next_state, reward, done, {}
import numpy as np

def value_iteration(env, gamma=0.99, theta=1e-6):
    value_table = np.zeros(env.n_states)
    policy = np.zeros(env.n_states, dtype=int)

    while True:
        delta = 0
        for state in range(env.n_states):
            v = value_table[state]
            q_values = np.zeros(env.n_actions)
            for action in range(env.n_actions):
                for prob, next_state, reward, done in env.P[state][action]:
                    q_values[action] += prob * (reward + gamma * value_table[next_state])
            value_table[state] = max(q_values)
            delta = max(delta, abs(v - value_table[state]))

        if delta < theta:
            break

    for state in range(env.n_states):
        q_values = np.zeros(env.n_actions)
        for action in range(env.n_actions):
            for prob, next_state, reward, done in env.P[state][action]:
                q_values[action] += prob * (reward + gamma * value_table[next_state])
        policy[state] = np.argmax(q_values)

    return policy, value_table

env = SimpleMDP()
policy, value_table = value_iteration(env)
print("Optimal Policy:", policy.reshape(env.size, env.size))
print("Value Table:", value_table.reshape(env.size, env.size))
