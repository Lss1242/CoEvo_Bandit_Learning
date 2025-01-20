import numpy as np

# Implementation of the Exp3 algorithm
class Exp3Player:
    def __init__(self, K, T):
        """_summary_

        **Parameters**:
        - K: Number of actions (arms)
        - T: Number of iterations
        - η_t (eta_t): Learning rate 
        - γ_t (gamma_t): Exploration probability
        """
        self.K = K
        self.T = T
        self.S = np.zeros(K)
        self.cumulative_reward = 0  # Overall cumulative reward
        
    
    def get_probabilities(self, t):
        """Calculate dynamic probabilities based on the current time step t"""
        
        #     debug
        #     print(f"self.S: {self.S.shape}")
        #     print(f"self.S: {self.S.max()}")
        #     print(f"self.S: {self.S.dtype}")
        
        # Use the parameter setting in Matrix Games with Bandit Feedback, 2021
        
        eta_t = np.sqrt(2 * np.log(self.K) / (t * self.K))
        gamma_t = min(np.sqrt(self.K * np.log(self.K) / t), 1)
        
        exp_values = np.exp(eta_t * self.S)
        probabilities = (1 - gamma_t) * (exp_values / np.sum(exp_values)) + gamma_t / self.K
        probabilities = np.clip(probabilities, 1e-10, 1.0)  # Prevent probabilities from being zero or NaN
        probabilities /= np.sum(probabilities)  # Re-normalize to ensure sum is 1
        
        return probabilities
    
    def select_action(self, probabilities):
        return np.random.choice(self.K, p=probabilities)
    
    def update(self, chosen_action, reward, probability):
        self.S[chosen_action] += (1 - reward) / probability
        # self.cumulative_reward[chosen_action] += reward
        self.cumulative_reward += reward  # Update overall cumulative reward
        
        

# Self-play function for the Exp3 algorithm

def play_exp3(matrix, K, T):
    player1 = Exp3Player(K, T)
    player2 = Exp3Player(K, T)
    
    p1_history = []
    p2_history = []
    cumulative_rewards = []
    
    for t in range(1, T + 1):
        try:
            p1 = player1.get_probabilities(t)
            p2 = player2.get_probabilities(t)
            
            i = player1.select_action(p1)
            j = player2.select_action(p2)
            
            r1 = matrix[i, j] + np.random.normal(0, 1) # Add Gaussian noise with mean 0 and variance 1
            r2 = -r1 
            
            player1.update(i, r1, p1[i])
            player2.update(j, r2, p2[j])
            
            p1_history.append(p1)
            p2_history.append(p2)
            cumulative_rewards.append(player1.cumulative_reward.copy())
        except FloatingPointError as e:
            print(f"FloatingPointError: {e}")
            continue
        
    return np.array(p1_history), np.array(p2_history), np.array(cumulative_rewards)

