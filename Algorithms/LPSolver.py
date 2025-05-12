# The following NE solver is adapted from the code: https://github.com/shuoyang2000/nash_equilibrium_solver/blob/main/solver.py and https://www.math.ucla.edu/~tom/gamesolve.html   

import numpy as np
from gurobipy import Model, GRB
import time

class NashEquilibriumSolver:
    def __init__(self, game_step: int, p1_action_num: int, p2_action_num: int, payoff_matrix: np.array):
        self.game_step = game_step
        self.action_choice_player1 = p1_action_num
        self.action_choice_player2 = p2_action_num
        self.action_total_num_player1 = self.action_choice_player1 ** game_step
        self.action_total_num_player2 = self.action_choice_player2 ** game_step
        self.payoff_matrix = payoff_matrix
        
        assert self.payoff_matrix.shape[0] == self.action_total_num_player1, "Payoff matrix row number does not match Player 1 pure strategy number"
        assert self.payoff_matrix.shape[1] == self.action_total_num_player2, "Payoff matrix column number does not match Player 2 pure strategy number"
    
    def solve_linear_program_player1(self, verbose: bool=False):
        start_time = time.time()
        LP_model = Model()
        LP_model.setParam('OutputFlag', 0)
        game_value = LP_model.addVar(name='game_value', vtype=GRB.CONTINUOUS)
        player1_actions = LP_model.addVars(self.action_total_num_player1, vtype=GRB.CONTINUOUS)
        LP_model.addConstr(player1_actions.sum() == 1)
        LP_model.addConstrs(player1_actions[i] >= 0 for i in range(self.action_total_num_player1))
        for player2_action_index in range(self.action_total_num_player2):
            lhs = 0
            for player1_action_index in range(self.action_total_num_player1):
                p1_utility = self.payoff_matrix[player1_action_index, player2_action_index]
                lhs += p1_utility * player1_actions[player1_action_index]
            LP_model.addConstr(lhs >= game_value)
        obj = game_value
        LP_model.setObjective(obj, GRB.MAXIMIZE)
        LP_model.optimize()
        if LP_model.Status == GRB.Status.OPTIMAL:
            player1_NE = LP_model.getAttr('x', player1_actions)
            filtered_player1_NE = {k: v for k, v in player1_NE.items() if v > 0.}
            NE_utility = LP_model.ObjVal
            if verbose:
                print("Player 1 strategy solving time: ", time.time() - start_time)
                print("Nash Equilibrium Profile Utility for Player 1: ", NE_utility)
                print("Nash Equilibrium Player 1 Strategy: ", filtered_player1_NE)
                print("------------------------------------------")
            return (filtered_player1_NE, NE_utility)
        else:
            print(LP_model.Status, GRB.Status.OPTIMAL)
            print("The linear programming for player 1 is infeasible, please double check.")
            return None
    
    def solve_linear_program_player2(self, verbose: bool=False):
        start_time = time.time()
        LP_model = Model()
        LP_model.setParam('OutputFlag', 0)
        game_value = LP_model.addVar(name='game_value', vtype=GRB.CONTINUOUS)
        player2_actions = LP_model.addVars(self.action_total_num_player2, vtype=GRB.CONTINUOUS)
        LP_model.addConstr(player2_actions.sum() == 1)
        LP_model.addConstrs(player2_actions[i] >= 0 for i in range(self.action_total_num_player2))
        for player1_action_index in range(self.action_total_num_player1):
            lhs = 0
            for player2_action_index in range(self.action_total_num_player2):
                p1_utility = self.payoff_matrix[player1_action_index, player2_action_index]
                lhs += p1_utility * player2_actions[player2_action_index]
            LP_model.addConstr(lhs <= game_value)
        obj = game_value
        LP_model.setObjective(obj, GRB.MINIMIZE)
        LP_model.optimize()
        if LP_model.Status == GRB.Status.OPTIMAL:
            player2_NE = LP_model.getAttr('x', player2_actions)
            filtered_player2_NE = {k: v for k, v in player2_NE.items() if v > 0.}
            NE_utility = LP_model.ObjVal
            if verbose:
                print("Player 2 strategy solving time: ", time.time() - start_time)
                print("Nash Equilibrium Profile Utility for Player 2: ", -NE_utility)
                print("Nash Equilibrium Player 2 Strategy: ", filtered_player2_NE)
                print("------------------------------------------")
            return (filtered_player2_NE, NE_utility)
        else:
            print(LP_model.Status, GRB.Status.OPTIMAL)
            print("The linear programming for player 2 is infeasible, please double check.")
            return None

