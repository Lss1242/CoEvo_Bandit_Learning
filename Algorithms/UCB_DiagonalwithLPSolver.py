import sys
import os

# Ensure path is correct
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
from Algorithms.LPSolver import NashEquilibriumSolver

class UCBPlayer:
    def __init__(self, K, T):
        self.K = K
        self.T = T
        self.cumulative_reward = 0  # Overall cumulative reward
        self.empirical_mean = np.zeros((K, K))  # \hat{A_{i,j}^t}
        self.action_counts = np.zeros((K, K))   # \n_{i,j}^t
        self.ucb_values = np.zeros((K, K))      # \tilde{A_{i,j}^t}
        self.previous_strategy = np.ones(K) / K  # Initialize with uniform distribution

    def maxmin(self, A):
        solver = NashEquilibriumSolver(game_step=1, p1_action_num=self.K, p2_action_num=self.K, payoff_matrix=A)
        result = solver.solve_linear_program_player1()
        if result:
            row_strategy, _ = result
            row_strategy = np.array([row_strategy.get(i, 0.0) for i in range(self.K)])
            # Clip values to ensure no zero probabilities, and normalize
            row_strategy = np.clip(row_strategy, 0, 1.0)
            row_strategy /= row_strategy.sum()
            # print("Selected row strategy: ", row_strategy)
            return row_strategy
        else:
            raise ValueError("No non-zero equilibrium found")

    def get_probabilities(self):
        probabilities = self.maxmin(A=self.ucb_values)
        return probabilities
    
    def select_action(self, probabilities):
        return np.random.choice(self.K, p=probabilities)
        
    def update_empirical_mean(self, i, j, reward):
        self.action_counts[i, j] += 1
        self.empirical_mean[i, j] = (self.empirical_mean[i, j] * (self.action_counts[i, j] - 1) + reward) / self.action_counts[i, j]
    
    def update(self, reward):
        self.cumulative_reward += reward
        
    def compute_ucb_values(self, t):
        for i in range(self.K):
            for j in range(self.K):
                if self.action_counts[i, j] > 0:
                    self.ucb_values[i, j] = self.empirical_mean[i, j] + np.sqrt(2 * np.log(2 * self.T**2 * self.K * self.K) / max(self.action_counts[i, j],1))
                else:
                    self.ucb_values[i, j] = 0 + np.sqrt(2 * np.log(2 * self.T**2 * self.K * self.K))  # The empirical mean is 0 if the action has not been selected yet

# Self-play function for the UCB algorithm

def play_ucb(matrix, K, T):
    player1 = UCBPlayer(K, T)
    player2 = UCBPlayer(K, T)
    
    p1_history = []
    p2_history = []
    cumulative_rewards = []

    for t in range(T):
        player1.compute_ucb_values(t)
        player2.compute_ucb_values(t)
        
        p1 = player1.get_probabilities()
        p2 = player2.get_probabilities()
        
        i = player1.select_action(p1)
        j = player2.select_action(p2)

        reward = matrix[i, j]+ np.random.normal(0, 1) # Add Gaussian noise with mean 0 and variance 1
        
        player1.update_empirical_mean(i, j, reward)
        player1.update(reward)
        # player1.compute_ucb_values(t)
        
        player2.update_empirical_mean(j, i, -reward)
        player2.update(-reward)
        # player2.compute_ucb_values(t)
        
        p1_history.append(p1)
        p2_history.append(p2)
        cumulative_rewards.append(player1.cumulative_reward.copy())
    
    return p1_history, p2_history, cumulative_rewards

# Example game matrix
# matrix = np.array([
#     [1, -1, 0],
#     [-1, 1, -1],
#     [0, -1, 1]
# ])
# matrix = np.array([[0, -1, -1, -1], [1, 0, 0, -1], [1, 0, 0, -1], [1, 1, 1, 0]]) # Diagonal matrix (draw)


# K = 4
# T = 3000

# p1_history, p2_history, cumulative_rewards = play_ucb(matrix, K, T)
# print("Player 1 history:", p1_history)
# print("Player 2 history:", p2_history)
# print("Cumulative rewards:", cumulative_rewards)

