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
        
    def train(self, episodes=1000):
        rewards = []
        for episode in range(episodes):
            self.env.newEpoch()
            state = self.get_state_key(self.env.getPosition(), self.env.getPackagesRemaining())
            total_reward = 0
            done = False
            
            while not done:
                action = self.choose_action(state)
                cell_type, new_pos, packages_left, done = self.env.takeAction(action)
                
                #reward function
                if cell_type > 0: #collected packages
                    reward = 10
                elif new_pos == self.env.getPosition(): #hit wall
                    reward = -1
                else:
                    reward = -1 #step penalty
                    
                new_state = self.get_state_key(now_pos, packages_left)
                self.update_q_table(state, action, reward, new_state)
                total_reward += reward
                state = new_state
                
            reward.append(total_reward)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
        return rewards
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, choices=['simple', 'multi', 'rgb'], required=True)
    parser.add_argument('--stochastic', action='store_true')
    args = parser.parse_args()
    
    agents = QLearningAgent(args.scenario, args.stochastic)
    rewards = args.train()
    
    #save final path and plot rewards
    agent.env.showPath(-1, savefig='final_path.jpg')
    
    #plotting code...
    
if __name__ == "__main__":
    main()