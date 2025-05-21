import numpy as np
import argparse
from FourRooms import FourRooms
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(self, scenario, stochastic=False):
        self.env = FourRooms(scenario, stochastic)
        self.actions = [FourRooms.UP, FourRooms.DOWN, FourRooms.LEFT, FourRooms.RIGHT]
        self.q_table = {} #key - (x, y, packages_left), values - [Q-values for each action]
        self.alpha = 0.2 #learning rate
        self.gamma = 0.9 #discount factor
        self.epsilon = 1.0 #initial exploration rate
        self.epsilon_decay = 0.998 #how quick exploration decreases
        self.epsilon_min = 0.01 #min exploration probability
        
        #for scenario 3 (RGB ordered collection)
        self.expected_order = [FourRooms.RED, FourRooms.GREEN, FourRooms.BLUE] if scenario == 'rgb' else None
        self.collected = [] #tracks collected packages for order validation
        
    def get_state_key(self, pos, package_left):
        return (*pos, package_left)
    
    def choose_action(self, state):
        """Select action using ε-greedy policy"""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions) #exploration
        else:
            return np.argmax(self.q_table.get(state, np.zeros(len(self.actions)))) #exploitation
        
    def update_q_table(self, state, action, reward, new_state):
        """Update Q-value using Bellman Equation"""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        if new_state not in self.q_table:
            self.q_table[new_state] = np.zeros(len(self.actions))
            
        best_next_action = np.argmax(self.q_table[new_state])
        td_target = reward + self.gamma * self.q_table[new_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
        
    def train(self, episodes=2000, exploration_strategy='decay'):
        """Train agent for given number of episodes/ephochs
        
        Args:
            episodes (int): Number of training episodes
            exploration_strategy (str): 'decay' or 'fixed' epsilon strategy
            
        Returns:
            list: Reward history of each episode"""
        rewards = []
        for episode in range(episodes):
            self.env.newEpoch()
            self.collected = [] #reset collected packages per epoch
            state = self.get_state_key(self.env.getPosition(), self.env.getPackagesRemaining())
            total_reward = 0
            done = False
            
            while not done:
                #get action, execute in environ
                action = self.choose_action(state)
                cell_type, new_pos, packages_left, done = self.env.takeAction(action)
                
                #scenario 3: check package order
                if self.expected_order and cell_type > 0:
                    if cell_type != self.expected_order[len(self.collected)]:
                        reward = -10
                        done = True
                    else:
                        reward = 10
                        self.collected.append(cell_type)
                else:
                    reward = -1 #step penalty
                
                #reward function
                if cell_type > 0: #collected packages
                    reward = 10
                elif new_pos == self.env.getPosition(): #hit wall
                    reward = -0.1
                else:
                    reward = -0.1 #step penalty
                
                #update Q-table and track rewards
                new_state = self.get_state_key(new_pos, packages_left)
                self.update_q_table(state, action, reward, new_state)
                total_reward += reward
                state = new_state
                
            rewards.append(total_reward)
            
            #update epsilon based on strategy
            if exploration_strategy == 'decay':
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            elif exploration_strategy == 'fixed':
                self.epsilon = max(0.05, 1.0 / (episode + 1)) #gradually reduce ε
                #pass #keep epsilon constant
            
            if episode % 100 == 0:
                print(f"Episode {episode}, Reward: {total_reward:.3f}, ε: {self.epsilon:.3f}")
                            
        return rewards
    
def smooth_rewards(rewards, window=50):
    return np.convolve(rewards, np.ones(window)/window, mode='valid')

def compare_exploration_strategies():
    #train with ε-decay
    agent_decay = QLearningAgent('simple')
    rewards_decay = agent_decay.train(exploration_strategy='decay')
    
    #train with fixed ε
    agent_fixed = QLearningAgent('simple')
    agent_fixed.epsilon = 0.1 #fixed exploration rate
    rewards_fixed = agent_fixed.train(exploration_strategy='fixed')
    
    #smooth rewards
    window = 25
    smooth_decay = smooth_rewards(rewards_decay, window)
    smooth_fixed = smooth_rewards(rewards_fixed, window)
    
    #plot results
    plt.figure(figsize=(10, 5))
    plt.plot(smooth_decay, label='ε-decay (0.998)')
    plt.plot(smooth_fixed, label='ε-fixed (1.0->0.05)')
    plt.xlabel('Episodes')
    plt.ylabel('Total Rewards')
    plt.title('Exploration Strategy Comparison (Scenario 1)')
    plt.legend()
    plt.savefig('exploration_comparison.jpg')
    plt.show()
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, choices=['simple', 'multi', 'rgb'], required=True)
    parser.add_argument('--stochastic', action='store_true')
    args = parser.parse_args()
    
    agent = QLearningAgent(args.scenario, args.stochastic)
    
    #rewards = agent.train()
    rewards = agent.train(episodes=2000)
    
    #save final path and plot rewards
    #agent.env.showPath(-1, savefig='final_path.jpg')
    agent.env.showPath(-1, savefig=f"final_path_{args.scenario}.jpg")
    
    
if __name__ == "__main__":
    """Uncomment to compare exploration strategies"""
    #compare_exploration_strategies()
    main()