import numpy as np
from typing import Tuple, Optional
from scipy import signal

from agents.game_utils import BoardPiece, PlayerAction, apply_player_action, PLAYER1, PLAYER2, initialize_game_state, connected_four, get_possible_moves
from agents.saved_state import SavedState


def generate_move_minimax(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState], depth: int = 4) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Generates the next move using the minimax algorithm.

    Parameters
    ----------
    board: np.ndarray
        Board to start with.

    player: BoardPiece
        The next player to make a move.

    saved_state: Optional[SavedState]
        Can be used to save a state of the game for future calculations. Not used here.

    depth: int
        Depth of the search tree to stop calculating.

    Returns
    -------
    :Tuple[PlayerAction, Optional[SavedState]]
        Tuple containing the move to play and the saved state.
    """
    if player == PLAYER1:
        evaluation = minimax_rec(0, depth, board, player, True)  # start maximizing if PLAYER1 is to play
    else:
        evaluation = minimax_rec(0, depth, board, player, False)  # start minimizing if PLAYER2 is to play
    return PlayerAction(evaluation[1]), None


def minimax_rec(current_depth: int, desired_depth: int, current_board: np.ndarray, player: BoardPiece,
                maximize: bool) -> (int, PlayerAction):
    """
    Recursive helper function for generate_move_minimax. Implements the minimax algorithm.
    
    Parameters
    ----------
    current_depth: int
        The current depth in the search tree.

    desired_depth: int
        The depth in the search tree to stop calculating.

    current_board: numpy.ndarray
        The current board of the game.

    player: BoardPiece
        The next player to make a move.

    maximize: bool
        True if the current player is maximizing, false if he is minimizing.

    Returns
    -------
    :(int, PlayerAction)
        Tuple of the evaluation after playing the move, which is also returned in this Tuple.
    """
    evaluations: [(int, PlayerAction)] = []  # to store all possible moves from this positions paired with their
    # evaluation
    possible_moves: [int] = get_possible_moves(current_board)
    if len(possible_moves) == 0 or current_depth == desired_depth:  # no more moves or desired depth reached -
        # recursion anchor
        evaluation: int = evaluate_position(current_board)
        return evaluation, -1  # -1 because we do not know the last played move - will be added when closing recursion

    for move in possible_moves:
        new_board = apply_player_action(current_board, move, player)
        evaluations.append(
            (minimax_rec(current_depth + 1, desired_depth, new_board, BoardPiece(3-player), not maximize)[0], move))  # change player and change between maximizing and minimizing
    if maximize:
        return max(evaluations, key=lambda x: x[0])  # get tuple with max evaluation
    else:
        return min(evaluations, key=lambda x: x[0])  # get tuple with min evaluation


def evaluate_position(board: np.ndarray) -> int:
    """
    Evaluates a board position. Use convolution to assess position (two pieces together are good, one piece near to
    an empty space is good, one piece next to a piece from the opponent is assessed as equal

    Parameters
    ----------
    board: numpy.ndarray
        Board to be evaluated.

    Returns
    -------
    :int
        The evaluation of the position.
    """
    if connected_four(board, PLAYER1):  # PLAYER1 won
        return 1_000_000_000_000
    if connected_four(board, PLAYER2):  # PLAYER2 won
        return -1_000_000_000_000

    evaluation_board = initialize_game_state()
    evaluation_board[board == PLAYER1] = 1  # set 1 for PLAYER1
    evaluation_board[board == PLAYER2] = -1  # set -1 for PLAYER2

    output: int = 0  # initial position evaluated as equal
    for snd in [[[1, 1]], [[1], [1]], np.identity(2), np.identity(2)[::-1, ::]]:  # setup of kernels
        output += int(sum(sum(signal.convolve2d(evaluation_board, snd, mode="valid"))))
    return output
