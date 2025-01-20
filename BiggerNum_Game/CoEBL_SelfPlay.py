import sys
import os

# Ensure path is correct
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from Algorithms.CoEBLv2 import play_coebl
from Algorithms.LPSolver import NashEquilibriumSolver
from BigNum_game import generate_bignum_game_matrix  # Import the BigNum game matrix generator

# Prequisite functions
def theoretical_regret(K, T):
    return 0.2 * np.sqrt(K**2 * np.arange(1, T + 1))

def total_variation_distance(p, q):
    return 0.5 * np.sum(np.abs(p - q))

def single_run(args):
    """A single run of the CoEBL self-play simulation."""    
    matrix, K, T, nash_strategy_p1, nash_strategy_p2, nash_utility_player1, seed = args
    
    np.random.seed(seed)
    
    p1_histories, p2_histories, cumulative_rewards = play_coebl(matrix, K, T)
    regrets = np.zeros(T)  # Preallocate regrets array
    tvd = np.zeros(T)      # Preallocate tvd array
    
    for t in range(T):
        current_strategy_p1 = p1_histories[t]
        current_strategy_p2 = p2_histories[t]

        # Calculate the total variation distance for both players
        tvd_p1 = total_variation_distance(current_strategy_p1, nash_strategy_p1)
        tvd_p2 = total_variation_distance(current_strategy_p2, nash_strategy_p2)

        # Store total variation distance and regret at time step t
        tvd[t] = tvd_p1 + tvd_p2
        regrets[t] = abs(nash_utility_player1 * (t + 1) - cumulative_rewards[t])
    
    return regrets, tvd

def self_play(matrix, K, T, num_runs=30):
    """Self-play function running multiple independent runs using multiprocessing."""
    
    # Compute Nash equilibria once
    nash_equilibrium_solver = NashEquilibriumSolver(game_step=1, p1_action_num=K, p2_action_num=K, payoff_matrix=matrix)
    nash_strategy_p1_original, nash_utility_player1 = nash_equilibrium_solver.solve_linear_program_player1()
    nash_strategy_p2, _ = nash_equilibrium_solver.solve_linear_program_player2()

    # Convert Nash strategies to numpy arrays once
    nash_strategy_p1 = np.array([nash_strategy_p1_original.get(i, 0.0) for i in range(K)])
    nash_strategy_p2 = np.array([nash_strategy_p2.get(i, 0.0) for i in range(K)])
    
    # Prepare the arguments for each run
    args = [(matrix, K, T, nash_strategy_p1, nash_strategy_p2, nash_utility_player1, seed) for seed in range(num_runs)]
    
    num_cores_to_use = min(50, cpu_count())
    print(f"Using {num_cores_to_use} CPU cores")
    # print(f"context: {cpu_count()} CPU cores available")
          
    with Pool(num_cores_to_use) as pool:
        results = pool.map(single_run, args)
    
    # Unpack results from parallel processing
    all_regrets, all_tvd = zip(*results)
    
    return nash_strategy_p1_original, nash_utility_player1, np.array(all_regrets), np.array(all_tvd)

# Example bignum game matrix
if __name__ == '__main__':
    nlist = [2, 3, 4, 5, 6, 7]  # List of different game sizes

    for n in nlist:
        # Generate bignum game matrix based on the size n
        matrix = generate_bignum_game_matrix(n)
        K = 2**n
        T = 3000  # Number of iterations

        # Run self-play for CoEBL algorithm
        nashpes, nash_utility_player1, regrets, tvd = self_play(matrix, K, T, num_runs=50)
        
        # Create output directory if it doesn't exist
        output_dir = './data_bignum/csv'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save results (regrets and TVD) to CSV files
        regrets_df = pd.DataFrame(regrets)
        tv_df = pd.DataFrame(tvd)

        regrets_df.to_csv(os.path.join(output_dir, f'coebl_selfplay_regrets_n{n}.csv'), index=False)
        tv_df.to_csv(os.path.join(output_dir, f'coebl_selfplay_tv_n{n}.csv'), index=False)
        
        # Display Nash equilibrium profile and utility for player 1
        print("Nash Equilibrium Profile: ",  np.array([nashpes.get(i, 0.0) for i in range(K)]))
        print("Nash Equilibrium Utility for Player 1: ", nash_utility_player1)
