import numpy as np
from typing import Tuple, Optional
from scipy import signal

import agents.game_utils
from agents.game_utils import *
from agents.saved_state import SavedState
from collections import defaultdict

FULL_BOARD: int = 0b0111111_0111111_0111111_0111111_0111111_0111111_0111111


def generate_move_minimax(board_player_one: int, board_player_two: int, player: BoardPiece,
                          saved_state: Optional[SavedState], depth: int = 7) -> Tuple[
    PlayerAction, Optional[SavedState]]:
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
    alpha: [int, [PlayerAction]] = [-1_000_000_000_000_000, [PlayerAction(-1)]]
    beta: [int, [PlayerAction]] = [1_000_000_000_000_000, [PlayerAction(-1)]]
    #dictio_test_one = defaultdict(dict)
    #dictio_test_two = defaultdict(dict)
    dictio_test_one = {-1: {}}
    dictio_test_two = {-1: {}}
    if player == PLAYER1:
        evaluation: list[int, [PlayerAction]] = \
            minimax_rec(0, depth, board_player_one, board_player_two, player, True, alpha,
                                 beta, dictio_test_one, [])  # start maximizing if PLAYER1 is to play
        #for key in dictio_test_one.keys():
        #    print(dictio_test_one[key], "---", bin(key))
    else:
        evaluation: list[int, [PlayerAction]] = \
            minimax_rec(0, depth, board_player_one, board_player_two, player, False, alpha,
                                 beta, dictio_test_two, [])  # start minimizing if PLAYER2 is to play
        #for key in dictio_test_two.keys():
        #    for key_two in dictio_test_two[key].keys():
        #        print(pretty_print_board(key, key_two),dictio_test_two[key][key_two])
    print(evaluation[0], " end: ", evaluation[1])
    return PlayerAction(evaluation[1][0]), None


def minimax_rec(current_depth: int, desired_depth: int, board_player_one: int, board_player_two: int,
                player: BoardPiece,
                maximize: bool, alpha: list[int, [PlayerAction]], beta: list[int, [PlayerAction]], dictionary: {}, moves_line: list[int]) -> list[int, [PlayerAction]]:
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
    possible_moves: [int] = get_possible_moves(board_player_one, board_player_two, player)
    if not possible_moves:
        current_game_state: GameState = check_end_state(board_player_one, board_player_two, 3-player)  # check for the last player
        if current_game_state == GameState.IS_WIN:
            #switch_negative = int(3-player == 2)  # last player was 2 and won, enable switch for negative value
            #evaluation: int = (-2*switch_negative+1) * int(1_000_000_000_000_000 * 10 ** (-current_depth))
            if player == PLAYER1:
                evaluation: int = int(-1_000_000_000_000_000_000 * 2 ** (-current_depth))
            else:
                evaluation: int = int(1_000_000_000_000_000_000 * 2 ** (-current_depth))
            #print(pretty_print_board(board_player_one, board_player_two), evaluation, player, moves_line)
            return [evaluation, moves_line]
        if current_game_state == GameState.IS_DRAW:
            return [0, moves_line]
    if current_depth == desired_depth:  #desired depth reached - recursion anchor
        evaluation: int = evaluate_position(board_player_one, board_player_two)
        return [evaluation, moves_line]
    if maximize:
        for move in possible_moves:
            new_board_player_one, new_board_player_two = apply_player_action(board_player_one, board_player_two, move,
                                                                             player)
            try:
                #raise KeyError
                saved_eval = dictionary[new_board_player_one][new_board_player_two]
                #print(moves_line," - ", saved_eval[1], " --- ", alpha)
                saved_eval[1] = moves_line + saved_eval[1]
                alpha = max([alpha, saved_eval], key=lambda x: x[0])
            except KeyError:
                moves_line_new = moves_line.copy()
                moves_line_new.append(move)
                recursion_eval = minimax_rec(current_depth + 1, desired_depth, new_board_player_one, new_board_player_two,
                                             BoardPiece(3 - player), not maximize,
                                             alpha, beta, dictionary, moves_line_new)
                alpha = max([alpha, recursion_eval], key=lambda x: x[0])
                #dictionary[new_board_player_one] = {new_board_player_two: [alpha[0], alpha[1][current_depth+1:]]}
            if beta[0] <= alpha[0]:
                return alpha
    else:
        for move in possible_moves:
            new_board_player_one, new_board_player_two = apply_player_action(board_player_one, board_player_two, move,
                                                                             player)
            try:
                #raise KeyError
                saved_eval = dictionary[new_board_player_one][new_board_player_two]
                #rint(moves_line," - ", saved_eval[1], " --- ", beta)
                saved_eval[1] = moves_line + saved_eval[1]
                beta = min([beta, saved_eval], key=lambda x: x[0])
            except KeyError:
                moves_line_new = moves_line.copy()
                moves_line_new.append(move)
                recursion_eval = minimax_rec(current_depth + 1, desired_depth, new_board_player_one, new_board_player_two,
                                             BoardPiece(3 - player), not maximize,
                                             alpha, beta, dictionary, moves_line_new)
                beta = min([beta, recursion_eval], key=lambda x: x[0])
                #dictionary[new_board_player_one] = {new_board_player_two: [beta[0], beta[1][current_depth+1:]]}
            if beta[0] <= alpha[0]:
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
    """not_played_tiles: int = FULL_BOARD - board_player_one - board_player_two
    not_played_and_player_one: int = not_played_tiles | board_player_one
    number_possible_connected_four_player_one: int = number_of_connected_n(not_played_and_player_one, 4)
    not_played_and_player_two: int = not_played_tiles | board_player_two
    number_possible_connected_four_player_two: int = number_of_connected_n(not_played_and_player_two, 4)
    return number_possible_connected_four_player_one - number_possible_connected_four_player_two"""


    output: int = 0  # initial position evaluated as equal
    p1_connected_two: int = number_of_connected_n(board_player_one, 2)
    p2_connected_two: int = number_of_connected_n(board_player_two, 2)
    connected_two_all_connection: int = number_of_connected_n(board_player_one | board_player_two, 2) - (
                p1_connected_two + p2_connected_two)
    output += (p1_connected_two * 2 - connected_two_all_connection) + (
                p2_connected_two * -2 + connected_two_all_connection)
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
        for _ in range(connected - 1):
            temp_board = temp_board & (temp_board >> i)
        out += bin(temp_board).count('1')
    return out
