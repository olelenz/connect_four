import numpy as np
from typing import Tuple, Optional
import random as rd

from agents.game_utils import BoardPiece, SavedState, PlayerAction, NO_PLAYER, apply_player_action, PLAYER1, PLAYER2


def generate_move(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState], depth: int) -> Tuple[
    PlayerAction, Optional[SavedState]]:
    evaluation = max_rec(0, depth, board, player)
    print("final eval is: "+str(evaluation))
    return PlayerAction(0), None


def min_rec(current_depth: int, desired_depth: int, current_board: np.ndarray, player: BoardPiece):
    evaluations: [(int, PlayerAction)] = []
    if current_depth == desired_depth:
        #calculate value for current position
        evaluation: int = evaluate_position(current_board)
        print("yay eval is: "+str(evaluation))
        return evaluation

    for move in get_possible_moves(current_board):
        new_board = apply_player_action(current_board, move, player)
        print("------------------------------------------")
        print("move to be played by "+str(player)+": "+str(move))
        if player == PLAYER1:
            evaluations.append((max_rec(current_depth + 1, desired_depth, new_board, PLAYER2)[0], move))
        else:
            evaluations.append((max_rec(current_depth + 1, desired_depth, new_board, PLAYER1)[0], move))
        print(" ")
    print(min(evaluations, key=lambda x:x[0])," herererererer")
    return min(evaluations)


def max_rec(current_depth: int, desired_depth: int, current_board: np.ndarray, player: BoardPiece):
    evaluations: [(int, PlayerAction)] = []
    if current_depth == desired_depth:
        # calculate value for current position
        evaluation: int = evaluate_position(current_board)
        print("uuh eval is: " + str(evaluation))
        return evaluation

    for move in get_possible_moves(current_board):
        print("------------------------------------------")
        print("move to be played by " + str(player) + ": " + str(move))
        new_board = apply_player_action(current_board, move, player)
        if player == PLAYER1:
            evaluations.append((min_rec(current_depth + 1, desired_depth, new_board, PLAYER2)[0], move))
        else:
            evaluations.append((min_rec(current_depth + 1, desired_depth, new_board, PLAYER1)[0], move))
        print(" ")
    print(max(evaluations, key=lambda x:x[0]), " aaaahhahahaahah")
    return max(evaluations)

def evaluate_position(board: np.ndarray) -> int:
    return rd.choices(range(-10,11))


def get_possible_moves(board: np.ndarray) -> [PlayerAction]:
    out: [PlayerAction] = []
    for i in range(0, 7):
        if board[5][i] == NO_PLAYER:
            out.append(PlayerAction(i))
    return out
