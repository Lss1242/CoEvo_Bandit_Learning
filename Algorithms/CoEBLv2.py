import sys
import os

# Ensure path is correct
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import numpy as np
from scipy.optimize import minimize
from Algorithms.LPSolver import NashEquilibriumSolver

# Define Gaussian Mutation operator
def gaussian_mutate(x, sigma):
    # y = x + sigma * np.random.randn()
    y = np.random.normal(x, sigma)
    return y

class CoEBLPlayer:
    def __init__(self, K, T):
        self.K = K
        self.T = T
        self.cumulative_reward = 0
        self.empirical_mean = np.zeros((K, K))  # \hat{A_{i,j}^t}
        self.action_counts = np.zeros((K, K))   # \n_{i,j}^t
        self.estimated_matrix = np.zeros((K, K))  # \tilde{A_{i,j}^t}

    def maxmin(self, A):
        solver = NashEquilibriumSolver(game_step=1, p1_action_num=self.K, p2_action_num=self.K, payoff_matrix=A)
        result = solver.solve_linear_program_player1()
        if result:
            row_strategy, _ = result
            row_strategy = np.array([row_strategy.get(i, 0.0) for i in range(self.K)])
            # Clip values to ensure no zero probabilities, and normalize
            row_strategy = np.clip(row_strategy, 0, 1.0)
            row_strategy /= row_strategy.sum()
            return row_strategy
        else:
            raise ValueError("No non-zero equilibrium found")

    def get_probabilities(self):
        probabilities = self.maxmin(A=self.estimated_matrix)
        return probabilities
    
    def select_action(self, probabilities):
        return np.random.choice(self.K, p=probabilities)
        
    def update_empirical_mean(self, i, j, reward):
        self.action_counts[i, j] += 1
        self.empirical_mean[i, j] = (self.empirical_mean[i, j] * (self.action_counts[i, j] - 1) + reward) / self.action_counts[i, j]
    
    def update(self, reward):
        self.cumulative_reward += reward
        
    def compute_estimated_matrix(self, t):
        for i in range(self.K):
            for j in range(self.K):
                # Use Gaussian mutation to explore the matrix
                self.estimated_matrix[i, j] = self.empirical_mean[i, j] + gaussian_mutate(np.sqrt(2 * np.log(2 * self.T**2 * self.K * self.K) / (max(self.action_counts[i, j],1) +1) )
                                                              , 1/(max(self.action_counts[i, j],1)))

    def fitness(self, strategy, matrix):
        """Compute the fitness of a strategy given a matrix."""
        min_payoff = float('inf')
        for y in np.eye(self.K):
            min_payoff = min(min_payoff, y.T @ matrix @ strategy)
        return min_payoff

def play_coebl(matrix, K, T):
    player1 = CoEBLPlayer(K, T)
    player2 = CoEBLPlayer(K, T)
    
    p1_history = []
    p2_history = []
    cumulative_rewards = []

    x_t = np.ones(K) / K  # Initial policy for player1
    for t in range(T):
        player1.compute_estimated_matrix(t)
        player2.compute_estimated_matrix(t)
        
        p1 = player1.get_probabilities()
        p2 = player2.get_probabilities()

        i = player1.select_action(p1)
        j = player2.select_action(p2)

        reward = matrix[i, j] + np.random.normal(0, 1) # Add Gaussian noise with mean 0 and variance 1

        player1.update_empirical_mean(i, j, reward)
        player1.update(reward)
        player2.update_empirical_mean(j, i, -reward)
        player2.update(-reward)
        
        # Line 6: Obtain a mutated policy x'
        x_prime = player1.maxmin(player1.estimated_matrix)
        
        # Line 7-11: Check if the fitness of the new policy is better than the current one
        if player1.fitness(x_prime, player1.estimated_matrix) > player1.fitness(x_t, player1.estimated_matrix):
            x_t = x_prime  # Update policy to new one


        p1_history.append(p1)
        p2_history.append(p2)
        cumulative_rewards.append(player1.cumulative_reward.copy())
    
    return p1_history, p2_history, cumulative_rewards

