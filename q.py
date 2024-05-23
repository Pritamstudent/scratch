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

def q_learning(env, num_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    q_table = np.zeros((env.n_states, env.n_actions))

    def epsilon_greedy_policy(state):
        if np.random.rand() < epsilon:
            return np.random.choice(env.n_actions)
        else:
            return np.argmax(q_table[state])

    for _ in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            action = epsilon_greedy_policy(state)
            next_state, reward, done, _ = env.step(action)
            best_next_action = np.argmax(q_table[next_state])
            td_target = reward + gamma * q_table[next_state, best_next_action]
            q_table[state, action] += alpha * (td_target - q_table[state, action])
            state = next_state

    policy = np.argmax(q_table, axis=1)
    return policy, q_table

env = SimpleMDP()
policy, q_table = q_learning(env)
print("Optimal Policy:", policy.reshape(env.size, env.size))
print("Q-Table:", q_table)
