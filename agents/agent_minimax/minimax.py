import numpy as np
from typing import Tuple, Optional
import random as rd

from scipy import signal

from agents.game_utils import BoardPiece, SavedState, PlayerAction, NO_PLAYER, apply_player_action, PLAYER1, PLAYER2, \
    initialize_game_state, connected_four


def generate_move_minimax(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState], depth: int = 4) -> Tuple[
    PlayerAction, Optional[SavedState]]:
    if player == PLAYER1:
        evaluation = max_rec(0, depth, board, player)
    else:
        evaluation = min_rec(0, depth, board, player)
    print("final eval is: " + str(evaluation))
    print("play the move: "+str(evaluation[1]))
    return PlayerAction(evaluation[1]), None


def min_rec(current_depth: int, desired_depth: int, current_board: np.ndarray, player: BoardPiece):
    evaluations: [(int, PlayerAction)] = []
    if current_depth == desired_depth:
        # calculate value for current position
        evaluation: [int] = evaluate_position(current_board, player)
        #print("yay eval is: "+str(evaluation))
        return evaluation

    for move in get_possible_moves(current_board):
        new_board = apply_player_action(current_board, move, player)
        #print("------------------------------------------")
        #print("move to be played by "+str(player)+": "+str(move))
        if player == PLAYER1:
            evaluations.append((max_rec(current_depth + 1, desired_depth, new_board, PLAYER2)[0], move))
        else:
            evaluations.append((max_rec(current_depth + 1, desired_depth, new_board, PLAYER1)[0], move))
        #print(" ")
    #print(min(evaluations, key=lambda x:x[0])," herererererer")
    return min(evaluations)


def max_rec(current_depth: int, desired_depth: int, current_board: np.ndarray, player: BoardPiece):
    evaluations: [(int, PlayerAction)] = []
    if current_depth == desired_depth:
        # calculate value for current position
        evaluation: [int] = evaluate_position(current_board, player)
        #print("uuh eval is: " + str(evaluation))
        return evaluation

    for move in get_possible_moves(current_board):
        #print("------------------------------------------")
        #print("move to be played by " + str(player) + ": " + str(move))
        new_board = apply_player_action(current_board, move, player)
        if player == PLAYER1:
            evaluations.append((min_rec(current_depth + 1, desired_depth, new_board, PLAYER2)[0], move))
        else:
            evaluations.append((min_rec(current_depth + 1, desired_depth, new_board, PLAYER1)[0], move))
        #print(" ")
    #print(max(evaluations, key=lambda x: x[0]), " aaaahhahahaahah")
    return max(evaluations)


def evaluate_position(board: np.ndarray, player: BoardPiece) -> [int]:
    output: int = 0
    if player == PLAYER2:
        if connected_four(board, PLAYER1): return[1_000_000_000_000]
        if connected_four(board, PLAYER2): return[-1_000_000_000_000]
    else:
        if connected_four(board, PLAYER2): return[-1_000_000_000_000]
        if connected_four(board, PLAYER1): return [1_000_000_000_000]
    evaluation_board = initialize_game_state()
    for row in range(6):
        for column in range(7):
            if board[row, column] == PLAYER1:
                evaluation_board[5 - row, column] = 1
            if board[row, column] == PLAYER2:
                evaluation_board[5 - row, column] = -1

    convolve_2nds = [[[1, 1]], [[1], [1]], np.identity(2), np.flip(np.identity(2), 1)]
    for snd in convolve_2nds:
        output += int(sum(sum(signal.convolve2d(evaluation_board, snd, mode="valid"))))
    return [output]


def get_possible_moves(board: np.ndarray) -> [PlayerAction]:
    out: [PlayerAction] = []
    for i in range(0, 7):
        if board[5][i] == NO_PLAYER:
            out.append(PlayerAction(i))
    return out
