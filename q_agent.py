import random

import numpy as np
from rl_base import Agent, Action, State
import os


class QAgent(Agent):

    def __init__(self, n_states, n_actions, name='QAgent', initial_q_value=0.0, q_table=None):
        super().__init__(name)

        self.learning_rate = 0.01
        self.gamma = 0.8
        self.epsilon = 0.5
        self.epsilon_decrement = 1e-5
        self.epsilon_minimum = 0.05

        self.action_space = [i for i in range(n_actions)]
        self.n_states = n_states
        self.q_table = q_table if q_table is not None else self.init_q_table(initial_q_value)

    def init_q_table(self, initial_q_value=0.):
        q_table = []
        for i in range(self.n_states):
            q_table.append([])
            for _ in self.action_space:
                q_table[i].append(initial_q_value)

        return q_table

    def update_action_policy(self) -> None:
        if self.epsilon > self.epsilon_minimum:
            self.epsilon = max(self.epsilon_minimum, self.epsilon - self.epsilon_decrement)

    def choose_action(self, state: State) -> Action:

        assert 0 <= state < self.n_states, \
            f"Bad state_idx. Has to be int between 0 and {self.n_states}"

        if random.random() < self.epsilon:
            return Action(random.choice(self.action_space))

        q_values = self.q_table[state]
        max_q_val = max(q_values)
        max_actions = []
        for i, val in enumerate(q_values):
            if val == max_q_val:
                max_actions.append(i)

        return Action(random.choice(max_actions)) 

    def learn(self, state: State, action: Action, reward: float, new_state: State, done: bool) -> None:
        current_q_val = self.q_table[state][action]
        
        max_q_val = max(self.q_table[new_state])
        
        target_q_val = reward + self.gamma + max_q_val
        
        self.q_table[state][action] += self.learning_rate * (target_q_val - current_q_val)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.q_table)

    def load(self, path):
        self.q_table = np.load(path)

    def get_instruction_string(self):
        return [f"Linearly decreasing eps-greedy: eps={self.epsilon:0.4f}"]
