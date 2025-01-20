import numpy as np
from scipy.optimize import minimize
from scipy.stats import entropy

# Implementation of Exp3-IX variant introduced in the paper:
# Uncoupled and Convergent Learning in Two-Player Zero-Sum Markov Games with Bandit Feedback, NeurIPS 2023

class Exp3_IXPlayer:
    def __init__(self, K, T):
        """
        Parameters:
        - K: Number of actions (arms)
        - T: Number of iterations
        - k_eta, k_beta, k_epsilon: Exponents for the learning rate
        """
        self.K = K
        self.T = T
        self.k_eta = 5 / 8
        self.k_beta = 3 / 8
        self.k_epsilon = 1 / 8
        self.cumulative_reward = 0  # Overall cumulative reward
    
    def get_probabilities(self, x_t, g_t, eta, t):
        def objective(x):
            return np.dot(x, g_t) + (1 / eta) * entropy(x, x_t)
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                       {'type': 'ineq', 'fun': lambda x: x - 1 / (self.K * t**2)})
        
        result = minimize(objective, x_t, constraints=constraints)
        return result.x if result.success else x_t

    def select_action(self, probabilities):
        return np.random.choice(self.K, p=probabilities)
    
    def update(self, chosen_action, reward, probabilities, beta, epsilon):
        g_t = np.zeros(self.K)
        for a in range(self.K):
            g_t[a] = (1 if chosen_action == a else 0) * reward / (probabilities[a] + beta) + epsilon * np.log(probabilities[a])
        self.cumulative_reward += reward  # Update overall cumulative reward
        return g_t, self.cumulative_reward
    
# Self-play function for the Exp3-IX algorithm
def play_exp3_ix(matrix, K, T):
    player1 = Exp3_IXPlayer(K, T)
    player2 = Exp3_IXPlayer(K, T)
    
    p1_history = []
    p2_history = []
    cumulative_rewards1 = []
    cumulative_rewards2 = []

    x_t1 = np.ones(K) / K
    x_t2 = np.ones(K) / K

    for t in range(1, T + 1):
        eta = t ** -player1.k_eta
        beta = t ** -player1.k_beta
        epsilon = t ** -player1.k_epsilon
        
        probabilities1 = x_t1  # Use x_t as probabilities directly
        # print("Probability for player 1: ", probabilities1)
        probabilities2 = x_t2  # Use x_t as probabilities directly
        # print("Probability for player 2: ", probabilities2)
                
        chosen_action1 = player1.select_action(probabilities1)
        chosen_action2 = player2.select_action(probabilities2)
        
        reward1 = matrix[chosen_action1, chosen_action2] + np.random.normal(0, 1) # Add Gaussian noise with mean 0 and variance 1
        reward2 = -reward1
        
        g_t1, cumulative_reward1 = player1.update(chosen_action1, reward1, probabilities1, beta, epsilon)
        g_t2, cumulative_reward2 = player2.update(chosen_action2, reward2, probabilities2, beta, epsilon)
        
        x_t1 = player1.get_probabilities(x_t1, g_t1, eta, t)
        x_t2 = player2.get_probabilities(x_t2, g_t2, eta, t)

        p1_history.append(probabilities1)
        p2_history.append(probabilities2)
        cumulative_rewards1.append(cumulative_reward1)
        cumulative_rewards2.append(cumulative_reward2)
        
    return np.array(p1_history), np.array(p2_history), np.array(cumulative_rewards1)

