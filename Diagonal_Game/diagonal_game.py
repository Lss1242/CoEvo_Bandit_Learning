# Generate the game instance for the Diagonal game

import numpy as np

def generate_diagonal_game_matrix(n):
    """
    Generates the payoff matrix for the Diagonal game given n.
    
    Parameters:
        n (int): The parameter n which defines the size of the strategy sets (U and V).
    
    Returns:
        np.ndarray: The payoff matrix for the Diagonal game.
    """
    num_strategies = 2**n
    matrix = np.zeros((num_strategies, num_strategies), dtype=int)
    
    for i in range(num_strategies):
        for j in range(num_strategies):
            u = bin(i).count('1')
            v = bin(j).count('1')
            if v < u:
                matrix[i, j] = 1
            elif v == u:
                matrix[i, j] = 0
            else:
                matrix[i, j] = -1
                
    return matrix

# Example usage
if __name__ == "__main__":
    n = 2
    matrix = generate_diagonal_game_matrix(n)
    print(matrix)

