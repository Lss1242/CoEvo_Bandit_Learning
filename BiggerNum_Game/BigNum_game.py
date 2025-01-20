import numpy as np

def generate_bignum_game_matrix(n):
    """
    Generates the payoff matrix for the BigNum game given n.
    
    Parameters:
        n (int): The parameter n which defines the size of the strategy sets (binary strings of length n).
    
    Returns:
        np.ndarray: The payoff matrix for the BigNum game.
    """
    num_strategies = 2**n
    matrix = np.zeros((num_strategies, num_strategies), dtype=int)
    
    for i in range(num_strategies):
        for j in range(num_strategies):
            if i == j:
                matrix[i, j] = 0
            elif i > j:
                matrix[i, j] = 1
            else:
                matrix[i, j] = -1
                
    return matrix

# Example usage
if __name__ == "__main__":
    n = 2
    matrix = generate_bignum_game_matrix(n)
    print(matrix)
