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