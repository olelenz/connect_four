import os
import platform
from typing import Tuple, Optional
from interruptingcow import timeout
import multiprocessing
import time

from agents.game_utils import *
from agents.saved_state import SavedState
from agents.agent_minimax.minimax_window_list import MINIMAX_EVALUATION_WINDOWS_LIST



FULL_BOARD: int = 0b0111111_0111111_0111111_0111111_0111111_0111111_0111111
START_VALUE: int = 100
MAX_VALUE: int = 1_000_000_000_000_000_000
MIN_MAX_FUNCTIONS = (min, max)
THREE_PIECES_IN_A_WINDOW_EVAL: int = 6

COLUMN_0_FILLED: int = 0b0111111_0000000_0000000_0000000_0000000_0000000_0000000
COLUMN_1_FILLED: int = 0b0000000_0111111_0000000_0000000_0000000_0000000_0000000
COLUMN_2_FILLED: int = 0b0000000_0000000_0111111_0000000_0000000_0000000_0000000
COLUMN_3_FILLED: int = 0b0000000_0000000_0000000_0111111_0000000_0000000_0000000
COLUMN_4_FILLED: int = 0b0000000_0000000_0000000_0000000_0111111_0000000_0000000
COLUMN_5_FILLED: int = 0b0000000_0000000_0000000_0000000_0000000_0111111_0000000
COLUMN_6_FILLED: int = 0b0000000_0000000_0000000_0000000_0000000_0000000_0111111

SHIFT_6_COLUMNS: int = 42
SHIFT_4_COLUMNS: int = 28
SHIFT_2_COLUMNS: int = 14


def generate_move_minimax_id(board_player_one: int, board_player_two: int, player: BoardPiece,
                             saved_state: Optional[SavedState], next_moves: list[int], depth: int = 8) -> list[
    int, [PlayerAction]]:
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

    alpha: [int, [PlayerAction]] = [-MAX_VALUE, [PlayerAction(-1)]]
    beta: [int, [PlayerAction]] = [MAX_VALUE, [PlayerAction(-1)]]
    dictio_one = {-1: {}}
    dictio_two = {-1: {}}
    use_mirror = use_mirror_functions(board_player_one, board_player_two)
    # use_mirror = False
    #print(minimax_data.__repr__())
    if player == PLAYER1:
        evaluation: list[int, [PlayerAction]] = \
            minimax_rec(depth, board_player_one, board_player_two, player, alpha,
                        beta, dictio_one, [], next_moves, 1, use_mirror)  # start maximizing if PLAYER1 is to play
    else:
        evaluation: list[int, [PlayerAction]] = \
            minimax_rec(depth, board_player_one, board_player_two, player, alpha,
                        beta, dictio_two, [], next_moves, 0, use_mirror)  # start minimizing if PLAYER2 is to play
    return evaluation


def generate_move_minimax(board_player_one: int, board_player_two: int, player: BoardPiece,
                          saved_state: Optional[SavedState], seconds: int = 5) -> Tuple[
    PlayerAction, Optional[SavedState]]:
    depth: int = 1
    move_output = multiprocessing.Value('i', -1)
    if platform.system() == "Windows":
        process_minimax = multiprocessing.Process(target=generate_move_loop_to_stop,
                                                  args=(move_output, board_player_one, board_player_two, player, depth))
        process_minimax.start()
        time.sleep(seconds)
        process_minimax.terminate()
        process_minimax.join()

        return move_output.value, None

    else:
        try:
            with timeout(seconds, exception=RuntimeError):
                generate_move_loop_to_stop(move_output, board_player_one, board_player_two, player, depth)
        except RuntimeError:
            pass
        return PlayerAction(move_output.value), None


def generate_move_loop_to_stop(move_output, board_player_one: int, board_player_two: int, player: BoardPiece, depth: int):
    evaluation: list[int, [PlayerAction]] = [0, []]
    while True:
        evaluation: list[int, [PlayerAction]] = generate_move_minimax_id(board_player_one, board_player_two,
                                                                         player, None, evaluation[1], depth)
        move_output.value = evaluation[1][0]
        print(depth, " moves: ", evaluation[1])
        if depth >= len(evaluation[1]) + 4:
            return
        depth += 1

def minimax_rec(current_depth: int, board_player_one: int, board_player_two: int,
                player: BoardPiece, alpha: list[int, [PlayerAction]], beta: list[int, [PlayerAction]], dictionary: {},
                moves_line: list[int], next_moves: list[int], minmax: int, use_mirror: bool) -> list[int, [PlayerAction]]:
    possible_moves, game_state = get_possible_moves_iterative((board_player_one, board_player_two, player), next_moves)
    if not possible_moves:
        return [handle_empty_moves_eval(player, game_state, current_depth), moves_line]
    if current_depth == 0:  # desired depth reached - recursion anchor
        evaluation: int = evaluate_board_using_windows(board_player_one, board_player_two)
        return [evaluation, moves_line]
    if minmax == 1:  # max
        for move in possible_moves:
            new_board_player_one, new_board_player_two = apply_player_action(
                (board_player_one, board_player_two, player), move)
            alpha = get_alpha(current_depth, new_board_player_one, new_board_player_two, player,
                              alpha, beta, dictionary, moves_line, next_moves, move, use_mirror)
            if beta[0] <= alpha[0]:
                return alpha
        return alpha
    else:  # min
        for move in possible_moves:
            new_board_player_one, new_board_player_two = apply_player_action(
                (board_player_one, board_player_two, player), move)
            beta = get_beta(current_depth, new_board_player_one, new_board_player_two, player, alpha, beta, dictionary,
                            moves_line, next_moves, move, use_mirror)
            if beta[0] <= alpha[0]:
                return beta
        return beta


def get_alpha(current_depth: int, new_board_player_one: int, new_board_player_two: int, player: BoardPiece, alpha: list[int, [PlayerAction]], beta: list[int, [PlayerAction]], dictionary: {}, moves_line: list[int], next_moves: list[int], move: PlayerAction, use_mirror: bool):
    saved_eval = get_eval_from_dictionary(new_board_player_one, new_board_player_two, dictionary, moves_line)
    if saved_eval is not None:
        return max([alpha, saved_eval], key=lambda x: x[0])

    moves_line_new = moves_line.copy()
    moves_line_new.append(move)
    recursion_eval = minimax_rec(current_depth - 1, new_board_player_one, new_board_player_two, BoardPiece(3 - player),
                                 alpha, beta, dictionary, moves_line_new, next_moves, 0, use_mirror)
    alpha = max([alpha, recursion_eval], key=lambda x: x[0])
    #dictionary[new_board_player_one] = {new_board_player_two: [alpha[0], alpha[1][-current_depth:]]}  # possible mistake here
    dictionary[new_board_player_one] = {new_board_player_two: [alpha[0], alpha[1]]}
    if use_mirror:
        add_mirrored_boards_to_dictionary(new_board_player_one, new_board_player_two, dictionary, alpha, current_depth)
    return alpha


def get_beta(current_depth: int, new_board_player_one: int, new_board_player_two: int, player: BoardPiece, alpha: list[int, [PlayerAction]], beta: list[int, [PlayerAction]], dictionary: {}, moves_line: list[int], next_moves: list[int], move: PlayerAction, use_mirror: bool):
    saved_eval = get_eval_from_dictionary(new_board_player_one, new_board_player_two, dictionary, moves_line)
    if saved_eval is not None:
        return min([beta, saved_eval], key=lambda x: x[0])

    moves_line_new = moves_line.copy()
    moves_line_new.append(move)
    recursion_eval = minimax_rec(current_depth - 1, new_board_player_one, new_board_player_two, BoardPiece(3 - player),
                                 alpha, beta, dictionary, moves_line_new, next_moves, 1, use_mirror)
    beta = min([beta, recursion_eval], key=lambda x: x[0])
    #dictionary[new_board_player_one] = {new_board_player_two: [beta[0], beta[1][-current_depth:]]}  # possible mistake here
    dictionary[new_board_player_one] = {new_board_player_two: [beta[0], beta[1]]}
    if use_mirror:
        add_mirrored_boards_to_dictionary(new_board_player_one, new_board_player_two, dictionary, beta, current_depth)
    return beta


def get_eval_from_dictionary(new_board_player_one: int, new_board_player_two: int, dictionary: {}, moves_line: [PlayerAction]) -> int | None:
    try:
        saved_eval = dictionary[new_board_player_one][new_board_player_two]
        #saved_eval[1] = moves_line + saved_eval[1]
        return saved_eval
    except KeyError:
        return None


def get_possible_moves_iterative(board_information: (int, int, BoardPiece), next_moves: list[int]) -> ([PlayerAction], GameState):
    try:
        return get_possible_moves(board_information[0], board_information[1], board_information[2], next_moves.pop(0))
    except IndexError:
        return get_possible_moves(board_information[0], board_information[1], board_information[2])


def handle_empty_moves_eval(player: BoardPiece, game_state: GameState, current_depth: int) -> int:
    if game_state == GameState.IS_WIN:
        if player == PLAYER1:
            return int(-START_VALUE * 2 ** (current_depth))
        else:
            return int(START_VALUE * 2 ** (current_depth))
    elif game_state == GameState.IS_DRAW:
        return 0
    else:
        raise AttributeError


def evaluate_position(board_player_one: int, board_player_two: int) -> int:
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


def evaluate_window(positions: [(int, int, int, int)], board_player1: int, board_player2: int) -> int:
    """
    Evaluates a single window, with emphasis on having 3 in a window and/or not sharing a window with pieces of the
    other player. Does not check for 4-connect as its checked elsewhere.

    Parameters
    ----------
    positions: [int]
        List of positions, represented as a board with a single piece on it.
    board_player1: int
        Board of player 1.
    board_player2: int
        Board of player 2.

    Returns
    -------
    :int
        Score of the window
    """
    counter_player1: int = 0
    counter_player2: int = 0
    for position in positions:  # count player pieces in the window
        if (position & board_player1) > 0:
            counter_player1 += 1
        elif (position & board_player2) > 0:
            counter_player2 += 1
    # return 0 if both player have pieces in the window, or both have none
    if (counter_player1 > 0 and counter_player2 > 0) or counter_player1 == counter_player2:
        return 0

    # putting more weight on 3 pieces in a window
    if counter_player1 == 3:
        return THREE_PIECES_IN_A_WINDOW_EVAL
    elif counter_player2 == 3:
        return -1 * THREE_PIECES_IN_A_WINDOW_EVAL
    return counter_player1**2 - counter_player2**2


def evaluate_board_using_windows(board_player1: int, board_player2: int) -> int:
    """
    Evaluates the board and returns a score.

    Parameters
    ----------
    board_player1: int
        Board of player 1.
    board_player2: int
        Board of player 2.

    Returns
    -------
    :int
        Evaluation of the board.

    """
    board_score = 0
    for window in MINIMAX_EVALUATION_WINDOWS_LIST:
        board_score += evaluate_window(window, board_player1, board_player2)
    return board_score


def mirror_boards(board_player1: int, board_player2: int) -> tuple[int, int]:
    """
    Mirrors the board by mirroring both player's board string

    Parameters
    ----------
    board_player1: int
        Board of player 1

    board_player2: int
        Board of player 2

    Returns
    -------
    tuple[int, int]:
        2 mirrored boards
    """
    return mirror_player_board(board_player1), mirror_player_board(board_player2)


def mirror_player_board(player_board) -> int:
    """
    Mirrors a single board around the middle column by bit shifting separate
    columns and putting them together.

    Parameters
    ----------
    player_board:
        The players board

    Returns
    -------
    int:
        The mirrored board
    """
    new_column_0: int = COLUMN_0_FILLED & (player_board << SHIFT_6_COLUMNS)
    # shifts the board to the left so that column 6 is in place
    # of column 0, then removes the other columns
    new_column_1: int = COLUMN_1_FILLED & (player_board << SHIFT_4_COLUMNS)
    new_column_2: int = COLUMN_2_FILLED & (player_board << SHIFT_2_COLUMNS)
    new_column_3: int = COLUMN_3_FILLED & player_board  # not shifted because in the middle
    new_column_4: int = COLUMN_4_FILLED & (player_board >> SHIFT_2_COLUMNS)
    new_column_5: int = COLUMN_5_FILLED & (player_board >> SHIFT_4_COLUMNS)
    new_column_6: int = COLUMN_6_FILLED & (player_board >> SHIFT_6_COLUMNS)
    return new_column_0 | new_column_1 | new_column_2 | new_column_3 | new_column_4 | new_column_5 | new_column_6  # puts all the columns together


def add_mirrored_boards_to_dictionary(board_player1: int, board_player2: int, dictionary, alpha_beta: list[int, [PlayerAction]], current_depth: int):  # TODO: refactor and test
    """
    Uses the mirror functions to add a mirrored board, its evaluation and mirrored playeractions to the dictionary.

    Parameters
    ----------
    board_player1: int
        Board player1
    board_player2: int
        Board player2
    dictionary: {}
        Dictionary  # should be reference of dictionary
    alpha_beta: tuple[int, int]
        Tuple that contains evaluation and playeractions
    current_depth: int
        Depth in the minimax algorithm

    """
    mirrored_board_player1, mirrored_board_player2 = mirror_boards(board_player1, board_player2)
    mirror_player_actions: Callable = np.vectorize(lambda arr: 6 - arr)  # mirrors each action in the move list
    mirrored_player_action = list(map(mirror_player_actions, alpha_beta[1]))
    dictionary[mirrored_board_player1] = {mirrored_board_player2: [alpha_beta[0], mirrored_player_action]}


def use_mirror_functions(board_player1: int, board_player2: int) -> bool:  # TODO: refactor and test
    """
    Checks if the board is symmetrical around the middle column. This is accomplished by selecting 2 columns,
    removing the other columns,  shifting them to the same position and using logical operations to evaluate if they
    are equal or not.
    This is done 3 times per player.

    Parameters
    ----------
    board_player1: int
        Board player1
    board_player2: int
        Board player2

    Returns
    -------
    bool:
        if mirror functions should be used in minimax or not
    """
    return (board_player1 & COLUMN_0_FILLED == (board_player1 << SHIFT_6_COLUMNS) & COLUMN_0_FILLED) and \
           (board_player1 & COLUMN_1_FILLED == (board_player1 << SHIFT_4_COLUMNS) & COLUMN_1_FILLED) and \
           (board_player1 & COLUMN_2_FILLED == (board_player1 << SHIFT_2_COLUMNS) & COLUMN_2_FILLED) and \
           (board_player2 & COLUMN_0_FILLED == (board_player2 << SHIFT_6_COLUMNS) & COLUMN_0_FILLED) and \
           (board_player2 & COLUMN_1_FILLED == (board_player2 << SHIFT_4_COLUMNS) & COLUMN_1_FILLED) and \
           (board_player2 & COLUMN_2_FILLED == (board_player2 << SHIFT_2_COLUMNS) & COLUMN_2_FILLED)

