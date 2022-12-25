import numpy as np
from typing import Tuple, Optional
from scipy import signal

import agents.game_utils
from agents.game_utils import BoardPiece, PlayerAction, apply_player_action, PLAYER1, PLAYER2, initialize_game_state, connected_four, get_possible_moves, create_dictionary_key, add_mirror_to_dictionary
from agents.saved_state import SavedState


def generate_move_minimax(board_player_one: int, board_player_two: int, player: BoardPiece, saved_state: Optional[SavedState], depth: int = 8) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Generates the next move using the minimax algorithm.

    Parameters
    ----------
    board_player_one: int
        Board PLAYER1.

    board_player_two: int
        Board PLAYER2.

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
    alpha: (int, PlayerAction) = (-1_000_000_000_000, PlayerAction(-1))
    beta: (int, PlayerAction) = (1_000_000_000_000, PlayerAction(-1))
    if player == PLAYER1:
        evaluation = minimax_rec(0, depth, board_player_one, board_player_two, player, True, alpha, beta, {})  # start maximizing if PLAYER1 is to play
    else:
        evaluation = minimax_rec(0, depth, board_player_one, board_player_two, player, False, alpha, beta, {})  # start minimizing if PLAYER2 is to play
    return PlayerAction(evaluation[1]), None


def minimax_rec(current_depth: int, desired_depth: int, board_player_one: int, board_player_two: int, player: BoardPiece,
                maximize: bool, alpha: (int, PlayerAction), beta: (int, PlayerAction), dictionary: {}) -> (int, PlayerAction):
    """
    Recursive helper function for generate_move_minimax. Implements the minimax algorithm.
    
    Parameters
    ----------
    current_depth: int
        The current depth in the search tree.

    desired_depth: int
        The depth in the search tree to stop calculating.

    board_player_one: int
        Board PLAYER1.

    board_player_two: int
        Board PLAYER2.

    player: BoardPiece
        The next player to make a move.

    maximize: bool
        True if the current player is maximizing, false if he is minimizing.

    alpha: (int, PlayerAction)
        Initial alpha value (very small) for alpha beta pruning and initial PlayerAction to be overwritten.

    beta: (int, PlayerAction)
        Initial beta value (very big) for alpha beta pruning and initial PlayerAction to be overwritten.

    dictionary: {}
        Transposition table
    Returns
    -------
    :(int, PlayerAction)
        Tuple of the evaluation after playing the move, which is also returned in this Tuple.
    """
    possible_moves: [int] = get_possible_moves(board_player_one, board_player_two)
    if len(possible_moves) == 0 or current_depth == desired_depth:  # no more moves or desired depth reached -
        # recursion anchor
        evaluation: int = evaluate_position(board_player_one, board_player_two, current_depth)
        return evaluation, -1  # -1 because we do not know the last played move - will be added when closing recursion

    if maximize:
        for move in possible_moves:
            new_board_player_one, new_board_player_two = apply_player_action(board_player_one, board_player_two, move, player)
            key = create_dictionary_key(new_board_player_one, new_board_player_two)
            try:
                check_alpha = dictionary[key]  # not sure if needed to throw KeyError, because if clause might not throw KeyError if false?
                if beta <= alpha:
                    return check_alpha
            except KeyError:
                alpha = max([alpha, (minimax_rec(current_depth + 1, desired_depth, new_board_player_one, new_board_player_two, BoardPiece(3-player), not maximize, alpha, beta, dictionary)[0], move)], key=lambda x: x[0])
                dictionary[key] = alpha
                add_mirror_to_dictionary(new_board_player_one, new_board_player_two, dictionary, alpha)
                if beta <= alpha:
                    return alpha
    else:
        for move in possible_moves:
            new_board_player_one, new_board_player_two = apply_player_action(board_player_one, board_player_two, move, player)
            key = create_dictionary_key(new_board_player_one, new_board_player_two)
            try:
                check_beta = dictionary[key]
                if beta <= alpha:
                    return check_beta
            except KeyError:
                beta = min([beta, (minimax_rec(current_depth + 1, desired_depth, new_board_player_one, new_board_player_two, BoardPiece(3 - player), not maximize, alpha, beta, dictionary)[
                    0], move)], key=lambda x: x[0])
                dictionary[key] = beta
                add_mirror_to_dictionary(new_board_player_one, new_board_player_two, dictionary, beta)
                if beta <= alpha:
                    return beta
    if maximize:
        return alpha
    else:
        return beta


def evaluate_position(board_player_one: int, board_player_two: int, depth: int = 0) -> int:
    """
    Evaluates a board position. Use convolution to assess position (two pieces together are good, one piece near to
    an empty space is good, one piece next to a piece from the opponent is assessed as equal

    Parameters
    ----------
    board_player_one: int
        Board PLAYER1.

    board_player_two: int
        Board PLAYER2.

    depth: int
        Depth at evaluation -> earlier wins are better.

    Returns
    -------
    :int
        The evaluation of the position.
    """
    if connected_four(board_player_one):  # PLAYER1 won
        return int(1_000_000_000_000 * 10**(-depth))
    if connected_four(board_player_two):  # PLAYER2 won
        return int(-1_000_000_000_000 * 10**(-depth))

    output: int = 0  # initial position evaluated as equal
    p1_connected_two: int = number_of_connected_n(board_player_one, 2)
    p2_connected_two: int = number_of_connected_n(board_player_two, 2)
    connected_two_all_connection: int = number_of_connected_n(board_player_one | board_player_two, 2) - (p1_connected_two + p2_connected_two)
    output += (p1_connected_two * 2 - connected_two_all_connection) + (p2_connected_two * -2 + connected_two_all_connection)
    return output


def number_of_connected_n(board: int, connected: int) -> int:
    """
    Evaluates the number of connected-n in a bitboard.

    Parameters
    ----------
    board: int
        The bitboard to be evaluated.

    connected: int
        The number of connected pieces to check for.

    Returns
    -------
    :int
        Number of connected-n.

    """
    assert connected > 0
    out: int = 0
    for i in [1, 6, 7, 8]:
        temp_board = board
        for _ in range(connected-1):
            temp_board = temp_board & (temp_board >> i)
        out += bin(temp_board).count('1')
    return out
