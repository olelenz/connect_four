import numpy as np
from typing import Tuple, Optional
import random as rd

from scipy import signal

from agents.game_utils import BoardPiece, SavedState, PlayerAction, NO_PLAYER, apply_player_action, PLAYER1, PLAYER2, \
    initialize_game_state, connected_four


def generate_move_minimax(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState], depth: int = 4) -> Tuple[
    PlayerAction, Optional[SavedState]]:
    if player == PLAYER1:
        evaluation = minimax_rec(0, depth, board, player, True)
    else:
        evaluation = minimax_rec(0, depth, board, player, False)
    print("final eval is: " + str(evaluation))
    print("play the move: "+str(evaluation[1]))
    return PlayerAction(evaluation[1]), None


def minimax_rec(current_depth: int, desired_depth: int, current_board: np.ndarray, player: BoardPiece, maximize: bool) -> [(int, PlayerAction)]:
    evaluations: [(int, PlayerAction)] = []
    possible_moves: [int] = get_possible_moves(current_board)

    if len(possible_moves) == 0 or current_depth == desired_depth:
        evaluation: int = evaluate_position(current_board)
        return [evaluation, -1]

    for move in possible_moves:
        new_board = apply_player_action(current_board, move, player)
        if player == PLAYER1:
            evaluations.append((minimax_rec(current_depth + 1, desired_depth, new_board, PLAYER2, not maximize)[0], move))
        else:
            evaluations.append((minimax_rec(current_depth + 1, desired_depth, new_board, PLAYER1, not maximize)[0], move))

    if maximize:
        return max(evaluations)
    else:
        return min(evaluations)


def evaluate_position(board: np.ndarray) -> int:
    output: int = 0
    if connected_four(board, PLAYER1): return 1_000_000_000_000
    if connected_four(board, PLAYER2): return -1_000_000_000_000
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
    return output


def get_possible_moves(board: np.ndarray) -> [PlayerAction]:
    out: [PlayerAction] = []
    if connected_four(board, PLAYER1) or connected_four(board, PLAYER2): return []
    for i in range(0, 7):
        if board[5][i] == NO_PLAYER:
            out.append(PlayerAction(i))
    return out
