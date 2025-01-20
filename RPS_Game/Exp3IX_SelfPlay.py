import sys
import os

# Ensure path is correct
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from Algorithms.Exp3_IX import play_exp3_ix
from Algorithms.LPSolver import NashEquilibriumSolver
from scipy.stats import entropy

def theoretical_regret(K, T):
    return 0.2 * np.sqrt(K**2 * np.arange(1, T + 1))

def kl_divergence(p, q):
    return entropy(p, q)

def single_run(args):
    """Run a single self-play instance."""
    matrix, K, T, nash_strategy_p1, nash_strategy_p2, nash_utility_player1, seed = args
    
    np.random.seed(seed)
    
    p1_histories, p2_histories, cumulative_rewards = play_exp3_ix(matrix, K, T)
    
    regrets = np.zeros(T)  # Preallocate regrets array
    kl_divergences = np.zeros(T)      # Preallocate kl_divergence array
    
    for t in range(T):
        current_strategy_p1 = p1_histories[t]
        current_strategy_p2 = p2_histories[t]

        # Calculate the total variation distance for both players
        kl_divergence_p1 = kl_divergence(current_strategy_p1, nash_strategy_p1)
        kl_divergence_p2 = kl_divergence(current_strategy_p2, nash_strategy_p2)

        # Store total variation distance and regret at time step t
        kl_divergences[t] = kl_divergence_p1 + kl_divergence_p2
        regrets[t] = abs(nash_utility_player1 * (t + 1) - cumulative_rewards[t])

    return regrets, kl_divergences

def self_play(matrix, K, T, num_runs=30):
    """Self-play function running multiple independent runs."""
    
    # Compute Nash equilibria once
    nash_equilibrium_solver = NashEquilibriumSolver(game_step=1, p1_action_num=K, p2_action_num=K, payoff_matrix=matrix)
    nash_strategy_p1_original, nash_utility_player1 = nash_equilibrium_solver.solve_linear_program_player1()
    nash_strategy_p2, _ = nash_equilibrium_solver.solve_linear_program_player2()

    # Convert Nash strategies to numpy arrays once
    nash_strategy_p1 = np.array([nash_strategy_p1_original.get(i, 0.0) for i in range(K)])
    nash_strategy_p2 = np.array([nash_strategy_p2.get(i, 0.0) for i in range(K)])
    
    # Prepare the arguments for each process
    args = [(matrix, K, T, nash_strategy_p1, nash_strategy_p2, nash_utility_player1, seed) for seed in range(num_runs)]
    
    num_cores_to_use = min(50, cpu_count())
    print(f"Using {num_cores_to_use} CPU cores")
    # print(f"context: {cpu_count()} CPU cores available")
          
    with Pool(num_cores_to_use) as pool:
        results = pool.map(single_run, args)
    
    # Unpack results from parallel processing
    all_regrets, all_kl_divergence = zip(*results)
    
    return nash_strategy_p1_original, nash_utility_player1, np.array(all_regrets), np.array(all_kl_divergence)

# Example Diagonal game matrix generation and running self-play experiments
if __name__ == '__main__':
        matrix = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])  # Rock-paper-scissors matrix

        K = 3
        T = 3000

        # Run self-play for Exp3-IX algorithm
        nashpes, nash_utility_player1, regrets, kl_divergences = self_play(matrix, K, T, num_runs=50)
        
        # Create output directory if it doesn't exist
        output_dir = './data_rps/csv'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save results (regrets and KL_DIVERGENCE) to CSV files
        regrets_df = pd.DataFrame(regrets)
        tv_df = pd.DataFrame(kl_divergences)

        regrets_df.to_csv(os.path.join(output_dir, f'exp3ix_selfplay_regrets.csv'), index=False)
        tv_df.to_csv(os.path.join(output_dir, f'exp3ix_selfplay_kl_divergences.csv'), index=False)
        
        # Display Nash equilibrium profile and utility for player 1
        print("Nash Equilibrium Profile: ",  np.array([nashpes.get(i, 0.0) for i in range(K)]))
        print("Nash Equilibrium Utility for Player 1: ", nash_utility_player1)

