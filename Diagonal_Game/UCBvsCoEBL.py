import sys
import os
from multiprocessing import Pool, cpu_count

# Ensure path is correct
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from Algorithms.UCB_DiagonalwithLPSolver import UCBPlayer
from Algorithms.CoEBLv2 import CoEBLPlayer
from Algorithms.LPSolver import NashEquilibriumSolver
from diagonal_game import generate_diagonal_game_matrix

# Pre-requisite functions

def theoretical_regret(K, T):
    return 0.3 * np.sqrt(K**2 * np.arange(1, T + 1))

def total_variation_distance(p, q):
    return 0.5 * np.sum(np.abs(p - q))


def play_UCBvsCoEBL(matrix, K, T):
    player1 = UCBPlayer(K, T)
    player2 = CoEBLPlayer(K, T)
    
    p1_history = []
    p2_history = []
    cumulative_rewards = []

    for t in range(1, T + 1):
        player1.compute_ucb_values(t)
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
        
        p1_history.append(p1)
        p2_history.append(p2)
        cumulative_rewards.append(player1.cumulative_reward.copy())

    return p1_history, p2_history, cumulative_rewards


def single_run(args):
    """Run a single compete instance."""
    
    matrix, K, T, nash_strategy_p1, nash_strategy_p2, nash_utility_player1, seed = args
    
    np.random.seed(seed)
    
    p1_histories, p2_histories, cumulative_rewards = play_UCBvsCoEBL(matrix, K, T)
    regrets = np.zeros(T)
    tvd = np.zeros(T)

    for t in range(T):
        current_strategy_p1 = p1_histories[t]
        current_strategy_p2 = p2_histories[t]

        tvd_p1 = total_variation_distance(current_strategy_p1, nash_strategy_p1)
        tvd_p2 = total_variation_distance(current_strategy_p2, nash_strategy_p2)

        tvd[t] = tvd_p1 + tvd_p2
        regrets[t] = nash_utility_player1 * (t + 1) - cumulative_rewards[t]

    return regrets, tvd

def compete(matrix, K, T, num_runs=30):
    """Compete function running multiple independent runs with multiprocessing."""
    
    nash_equilibrium_solver = NashEquilibriumSolver(game_step=1, p1_action_num=K, p2_action_num=K, payoff_matrix=matrix)
    nash_strategy_p1_original, nash_utility_player1 = nash_equilibrium_solver.solve_linear_program_player1()
    nash_strategy_p2_original, nash_utility_player2 = nash_equilibrium_solver.solve_linear_program_player2()

    nash_strategy_p1 = np.array([nash_strategy_p1_original.get(i, 0.0) for i in range(K)])
    nash_strategy_p2 = np.array([nash_strategy_p2_original.get(i, 0.0) for i in range(K)])

    # Prepare the arguments for each run
    args = [(matrix, K, T, nash_strategy_p1, nash_strategy_p2, nash_utility_player1, seed) for seed in range(num_runs)]

    # Number of cores to use
    num_cores_to_use = min(50, cpu_count())
    print(f"Using {num_cores_to_use} CPU cores")

    with Pool(num_cores_to_use) as pool:
        results = pool.map(single_run, args)

    all_regrets, all_tvd = zip(*results)

    return nash_strategy_p1_original, nash_utility_player1, np.array(all_regrets), np.array(all_tvd)
    
 
# Example Diagonal game matrix
if __name__ == '__main__':
    nlist = [2, 3, 4, 5, 6, 7, 8]  # List of different game sizes

    for n in nlist:
        matrix = generate_diagonal_game_matrix(n)
        K = 2**n
        T = 3000

        # Run the self-play experiments using multiprocessing
        nashpes, nash_utility_player1, regrets, tvd = compete(matrix, K, T, num_runs=50)

        # Save regrets and TVD to CSV files
        output_dir = './data/csv'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        regrets_df = pd.DataFrame(regrets)
        tv_df = pd.DataFrame(tvd)

        regrets_df.to_csv(os.path.join(output_dir, f'ucb_vs_coebl_regrets_n{n}.csv'), index=False)
        tv_df.to_csv(os.path.join(output_dir, f'ucb_vs_coebl_tv_n{n}.csv'), index=False)


        print("Nash Equilibrium Profile: ", np.array([nashpes.get(i, 0.0) for i in range(K)]))
        print("Nash Equilibrium Utility for Player 1: ", nash_utility_player1)

