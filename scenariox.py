import numpy as np
import argparse
from FourRooms import FourRooms

class QLearningAgent:
    def __init__(self, scenario, stochastic=False):
        self.env = FourRooms(scenario, stochastic)
        self.actions = [FourRooms.UP, FourRooms.DOWN, FourRooms.LEFT, FourRooms.RIGHT]
        self.q_table = {} #key - (x, y, packages_left), values - [Q-values for each action]
        self.alpha = 0.1 #learning rate
        self.gamma = 0.9 #discount factor
        self.epsilon = 1.0 #initial exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
    def get_state_key(self, pos, package_left):
        return (*pos, package_left)
    
    def choose_action(self, state):
        if np.rand() < self.epsilon:
            return np.random.choice(self.action) #exploration
        else:
            return np.argmax(self.q_table.get(state, np.zeros(len(self.actions)))) #exploitation
        
    def update_q_table(self, state, action, reward, new_state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        if new_state not in self.q_table:
            self.q_table[new_state] = np.zeros(len(self.actions))
            
        best_next_action = np.argmax(self.q_table[new_state])
        td_target = reward + self.gamma * self.q_table[new_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error