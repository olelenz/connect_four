import os
import platform
from typing import Tuple, Optional
from interruptingcow import timeout
import multiprocessing
import time
import multiprocessing.sharedctypes

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


def generate_move_minimax(board_player_one: int, board_player_two: int, player: BoardPiece,
                          saved_state: Optional[SavedState], seconds: int = 5) -> Tuple[
    PlayerAction, Optional[SavedState]]:
    """
    Starting point to use the minimax algorithm. Handles the interrupting after the amount of seconds given.
    Parameters
    ----------
    board_player_one: int
        Board player one.
    board_player_two: int
        Board player two.
    player: BoardPiece
        The next player to make a move.
    saved_state: Optional[SavedState]
        Can be used to save a state of the game for future calculations. Not used here.
    seconds: int
        Time given for minimax-calculation.

    Returns
    -------
    :Tuple[PlayerAction, Optional[SavedState]]
        Tuple containing the move to play and the saved state.
    """
    depth: int = 1
    move_output: multiprocessing.sharedctypes.Synchronized = multiprocessing.Value('i', -1)
    if platform.system() == "Windows":  # Platform dependent, full test coverage problematic.
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


def generate_move_loop_to_stop(move_output: multiprocessing.sharedctypes.Synchronized, board_player_one: int, board_player_two: int, player: BoardPiece, depth: int):
    """

    Parameters
    ----------
    move_output: multiprocessing.sharedctypes.Synchronized
        Variable to return from this function - needed for execution in another process.
    board_player_one: int
        Board player one.
    board_player_two: int
        Board player two.
    player: BoardPiece
        The player to make a move.
    depth: int
        Current depth (decreasing).

    Returns
    -------

    """
    evaluation: list[int, [PlayerAction]] = [0, []]
    while True:
        evaluation: list[int, [PlayerAction]] = generate_move_minimax_id(board_player_one, board_player_two,
                                                                         player, None, evaluation[1], depth)
        move_output.value = evaluation[1][0]
        print(depth, " moves: ", evaluation[1])
        if depth >= len(evaluation[1]) + 4:
            return
        depth += 1


def generate_move_minimax_id(board_player_one: int, board_player_two: int, player: BoardPiece,
                             saved_state: Optional[SavedState], next_moves: list[int], depth: int = 8) -> list[
    int, [PlayerAction]]:
    """
    Generates the next move using the minimax algorithm.
    Parameters
    ----------
    board_player_one: int
        Board player one.
    board_player_two: int
        Board player two.
    player: BoardPiece
        The next player to make a move.
    saved_state: Optional[SavedState]
        Can be used to save a state of the game for future calculations. Not used here.
    next_moves: list[int]
        Move order to try first to improve alpha-beta-pruning.
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
    if player == PLAYER1:
        evaluation: list[int, [PlayerAction]] = \
            minimax_rec(depth, board_player_one, board_player_two, player, alpha,
                        beta, dictio_one, [], next_moves, 1, use_mirror)  # start maximizing if PLAYER1 is to play
    else:
        evaluation: list[int, [PlayerAction]] = \
            minimax_rec(depth, board_player_one, board_player_two, player, alpha,
                        beta, dictio_two, [], next_moves, 0, use_mirror)  # start minimizing if PLAYER2 is to play
    return evaluation


def minimax_rec(current_depth: int, board_player_one: int, board_player_two: int,
                player: BoardPiece, alpha: list[int, [PlayerAction]], beta: list[int, [PlayerAction]], dictionary: {},
                moves_line: list[int], next_moves: list[int], minmax: int, use_mirror: bool) -> list[int, [PlayerAction]]:
    """
    Main recursion function for the minimax algorith. Handles the anchors and the calls to further needed calculation.
    Parameters
    ----------
    current_depth: int
        Current depth (decreasing).
    board_player_one: int
        Board player one.
    board_player_two: int
        Board player two.
    player: BoardPiece
        The player to make a move.
    alpha: list[int, [PlayerAction]]
        Alpha for alpha-beta pruning.
    beta: list[int, [PlayerAction]]
        Beta for alpha-beta pruning.
    dictionary: {}
        Dictionary for the transposition table.
    moves_line: list[int]
        Current line of move taken.
    next_moves: list[int]
        Next moves to evaluate first from recent calculation for better pruning.
    minmax: int
        Flag for min (0) and max (1) of the algorithm.
    use_mirror: bool
        If the mirrored board should be saved in the transposition table.

    Returns
    -------
    :list[int, [PlayerAction]]
        List containing the evaluation and the list of PlayerActions to get to that evaluation.
    """
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


def get_alpha(current_depth: int, board_player_one: int, board_player_two: int, player: BoardPiece, alpha: list[int, [PlayerAction]], beta: list[int, [PlayerAction]], dictionary: {}, moves_line: list[int], next_moves: list[int], move: PlayerAction, use_mirror: bool) -> list[int, [PlayerAction]]:
    """
    Function to calculate the new alpha-value and then continue in the recursion.
    Parameters
    ----------
    current_depth: int
        Current depth (decreasing).
    board_player_one: int
        Board player one.
    board_player_two: int
        Board player two.
    player: BoardPiece
        The player to make a move.
    alpha: list[int, [PlayerAction]]
        Alpha for alpha-beta pruning.
    beta: list[int, [PlayerAction]]
        Beta for alpha-beta pruning.
    dictionary: {}
        Dictionary for the transposition table.
    moves_line: list[int]
        Current line of move taken.
    next_moves: list[int]
        Next moves to evaluate first from recent calculation for better pruning.
    move: PlayerAction
        The last played moved.
    use_mirror: bool
        If the mirrored board should be saved in the transposition table.
    Returns
    -------
    :list[int, [PlayerAction]]
        Values for alpha.
    """
    saved_eval = get_eval_from_dictionary(board_player_one, board_player_two, dictionary)
    if saved_eval is not None:  # There is an entry in the transposition table.
        return max([alpha, saved_eval], key=lambda x: x[0])

    moves_line_new = moves_line.copy()
    moves_line_new.append(move)
    recursion_eval = minimax_rec(current_depth - 1, board_player_one, board_player_two, BoardPiece(3 - player),
                                 alpha, beta, dictionary, moves_line_new, next_moves, 0, use_mirror)
    alpha = max([alpha, recursion_eval], key=lambda x: x[0])
    dictionary[board_player_one] = {board_player_two: [alpha[0], alpha[1]]}
    if use_mirror:
        add_mirrored_boards_to_dictionary(board_player_one, board_player_two, dictionary, alpha, current_depth)
    return alpha


def get_beta(current_depth: int, board_player_one: int, board_player_two: int, player: BoardPiece, alpha: list[int, [PlayerAction]], beta: list[int, [PlayerAction]], dictionary: {}, moves_line: list[int], next_moves: list[int], move: PlayerAction, use_mirror: bool) -> list[int, [PlayerAction]]:
    """
    Function to calculate the new beta-value and then continue in the recursion.
    Parameters
    ----------
    current_depth: int
        Current depth (decreasing).
    board_player_one: int
        Board player one.
    board_player_two: int
        Board player two.
    player: BoardPiece
        The player to make a move.
    alpha: list[int, [PlayerAction]]
        Alpha for alpha-beta pruning.
    beta: list[int, [PlayerAction]]
        Beta for alpha-beta pruning.
    dictionary: {}
        Dictionary for the transposition table.
    moves_line: list[int]
        Current line of move taken.
    next_moves: list[int]
        Next moves to evaluate first from recent calculation for better pruning.
    move: PlayerAction
        The last played moved.
    use_mirror: bool
        If the mirrored board should be saved in the transposition table.
    Returns
    -------
    :list[int, [PlayerAction]]
        Values for beta.
    """
    saved_eval = get_eval_from_dictionary(board_player_one, board_player_two, dictionary)
    if saved_eval is not None:
        return min([beta, saved_eval], key=lambda x: x[0])

    moves_line_new = moves_line.copy()
    moves_line_new.append(move)
    recursion_eval = minimax_rec(current_depth - 1, board_player_one, board_player_two, BoardPiece(3 - player),
                                 alpha, beta, dictionary, moves_line_new, next_moves, 1, use_mirror)
    beta = min([beta, recursion_eval], key=lambda x: x[0])
    dictionary[board_player_one] = {board_player_two: [beta[0], beta[1]]}
    if use_mirror:
        add_mirrored_boards_to_dictionary(board_player_one, board_player_two, dictionary, beta, current_depth)
    return beta


def get_eval_from_dictionary(board_player_one: int, board_player_two: int, dictionary: {}) -> [int, [int]] or None:
    """
    Function to get the value from the transposition table corresponding to the given boards.
    Parameters
    ----------
    board_player_one: int
        Board player one.
    board_player_two: int
        Board player two.
    dictionary: {}
        Transposition table.
    Returns
    -------
    : [int, [int]] or None
        None if there is no entry, otherwise the entry.
    """
    try:
        print(dictionary[board_player_one][board_player_two])
        return dictionary[board_player_one][board_player_two]
    except KeyError:
        return None


def get_possible_moves_iterative(board_information: (int, int, BoardPiece), next_moves: list[int]) -> ([PlayerAction], GameState):
    """
    Function to get the possible moves. Distinguishes between a function call with and without the use of the better move-ordering.
    Parameters
    ----------
    board_information: (int, int, BoardPiece)
        Tuple containing the two current boards and the current player.
    next_moves: list[int]
        List containing the next moves for better ordering.
    Returns
    -------
    :([PlayerAction], GameState)
        Tuple with the list of playeractions and the current GameState.
    """
    try:
        return get_possible_moves(board_information[0], board_information[1], board_information[2], next_moves.pop(0))
    except IndexError:
        return get_possible_moves(board_information[0], board_information[1], board_information[2])


def handle_empty_moves_eval(player: BoardPiece, game_state: GameState, current_depth: int) -> int:
    """
    Function to handle the recursion anchor when there are no more moves possible to play.
    Parameters
    ----------
    player: BoardPiece
        Current player.
    game_state: GameState
        Current state of the game.
    current_depth: int
        Current depth (decreasing).

    Raises
    ----------
    AttributeError
        If the current game state is STILL_PLAYING - this function should not be called because there are still moves to be played.

    Returns
    -------
    :int
        Evaluation of the position.
    """
    if game_state == GameState.IS_WIN:
        if player == PLAYER1:
            return int(-START_VALUE * 2 ** current_depth)
        else:
            return int(START_VALUE * 2 ** current_depth)
    elif game_state == GameState.IS_DRAW:
        return 0
    else:
        raise AttributeError


def evaluate_window(window_positions: [(int, int, int, int)], board_player_one: int, board_player_two: int) -> int:
    """
    Evaluates a single window, with emphasis on having 3 in a window and/or not sharing a window with pieces of the
    other player. Does not check for 4-connect as its checked elsewhere.

    Parameters
    ----------
    window_positions: [int]
        List of positions, represented as a board with a single piece on it.
    board_player_one: int
        Board of player one.
    board_player_two: int
        Board of player one.

    Returns
    -------
    :int
        Evaluation of the window.
    """
    number_of_player_one_pieces: int = 0  # could be one tuple, but makes code less readable?
    number_of_player_two_pieces: int = 0
    for position in window_positions:  # Counts player pieces in the window. TODO: more efficient way for this?
        if (position & board_player_one) > 0:
            number_of_player_one_pieces += 1
        elif (position & board_player_two) > 0:
            number_of_player_two_pieces += 1
    return calculate_evaluation_score(number_of_player_one_pieces, number_of_player_two_pieces)


def calculate_evaluation_score(number_of_player_one_pieces: int, number_of_player_two_pieces: int) -> int:
    """

    Parameters
    ----------
    number_of_player_one_pieces: int
        Amount of player one pieces in the current window.
    number_of_player_two_pieces: int
        Amount of player one pieces in the current window.

    Returns
    -------
    int:
        Evaluation of the window.
    """
    # Returns zero if both players have pieces in the window, or both have none.
    if (number_of_player_one_pieces > 0 and number_of_player_two_pieces > 0) or \
            number_of_player_one_pieces == number_of_player_two_pieces:
        return 0
    # Returns currently 6 points if a player has 3 pieces in a window. Amount of points is managed in a global variable.
    elif number_of_player_one_pieces == 3:
        return THREE_PIECES_IN_A_WINDOW_EVAL
    elif number_of_player_two_pieces == 3:
        return -1 * THREE_PIECES_IN_A_WINDOW_EVAL
    else:
        return number_of_player_one_pieces ** 2 - number_of_player_two_pieces ** 2
        # Returns 1 or 4 points depending on the amount of pieces to a player.
        # Player two gets negative points.


def evaluate_board_using_windows(board_player_one: int, board_player_two: int) -> int:
    """
    Evaluates the board and returns a score.

    Parameters
    ----------
    board_player_one: int
        Board of player 1.
    board_player_two: int
        Board of player 2.

    Returns
    -------
    :int
        Evaluation of the board.
    """
    board_evaluation: int = 0
    for window in MINIMAX_EVALUATION_WINDOWS_LIST:
        board_evaluation += evaluate_window(window, board_player_one, board_player_two)
    return board_evaluation


def mirror_boards(board_player_one: int, board_player_two: int) -> tuple[int, int]:
    """
    Mirrors the board by mirroring both player's board string.

    Parameters
    ----------
    board_player_one: int
        Board of player one.

    board_player_two: int
        Board of player two.

    Returns
    -------
    tuple[int, int]:
        Two mirrored boards.
    """
    return mirror_player_board(board_player_one), mirror_player_board(board_player_two)


def mirror_player_board(player_board) -> int:
    """
    Mirrors a single board around the middle column by bit shifting separate
    columns and putting them together.

    Parameters
    ----------
    player_board:
        The players board.

    Returns
    -------
    int:
        The mirrored board.
    """
    new_column_0: int = COLUMN_0_FILLED & (player_board << SHIFT_6_COLUMNS)
    # Shifts the board to the left so that column six is in place
    # of column zero, then removes the other columns.
    new_column_1: int = COLUMN_1_FILLED & (player_board << SHIFT_4_COLUMNS)
    new_column_2: int = COLUMN_2_FILLED & (player_board << SHIFT_2_COLUMNS)
    new_column_3: int = COLUMN_3_FILLED & player_board  # not shifted because in the middle
    new_column_4: int = COLUMN_4_FILLED & (player_board >> SHIFT_2_COLUMNS)
    new_column_5: int = COLUMN_5_FILLED & (player_board >> SHIFT_4_COLUMNS)
    new_column_6: int = COLUMN_6_FILLED & (player_board >> SHIFT_6_COLUMNS)
    return new_column_0 | new_column_1 | new_column_2 | new_column_3 | new_column_4 | new_column_5 | new_column_6
    # Puts all the columns together.


def add_mirrored_boards_to_dictionary(board_player_one: int, board_player_two: int, dictionary, alpha_beta: list[int, [PlayerAction]], current_depth: int):  # TODO: refactor and test
    """
    Uses the mirror functions to add a mirrored board, its evaluation and mirrored playeractions to the dictionary.

    Parameters
    ----------
    board_player_one: int
        Board player one.
    board_player_two: int
        Board player two.
    dictionary: {}
        Dictionary.  # should be reference of dictionary
    alpha_beta: tuple[int, int]
        Tuple that contains evaluation and playeractions.
    current_depth: int
        Depth in the minimax algorithm.
    """
    mirrored_board_player_one, mirrored_board_player_two = mirror_boards(board_player_one, board_player_two)
    mirror_player_actions: Callable = np.vectorize(lambda arr: 6 - arr)  # mirrors each action in the move list
    mirrored_player_actions = list(map(mirror_player_actions, alpha_beta[1]))
    dictionary[mirrored_board_player_one] = {mirrored_board_player_two: [alpha_beta[0], mirrored_player_actions]}


def use_mirror_functions(board_player_one: int, board_player_two: int) -> bool:
    """
    Checks if the board is symmetrical around the middle column. This is accomplished by selecting two columns,
    removing the other columns, shifting them to the same position and using logical operations to evaluate if they
    are equal or not.
    This is done three times per player.

    Parameters
    ----------
    board_player_one: int
        Board player one.
    board_player_two: int
        Board player two.

    Returns
    -------
    bool:
        If mirror functions should be used in minimax or not.
    """
    return (board_player_one & COLUMN_0_FILLED == (board_player_one << SHIFT_6_COLUMNS) & COLUMN_0_FILLED) and \
           (board_player_one & COLUMN_1_FILLED == (board_player_one << SHIFT_4_COLUMNS) & COLUMN_1_FILLED) and \
           (board_player_one & COLUMN_2_FILLED == (board_player_one << SHIFT_2_COLUMNS) & COLUMN_2_FILLED) and \
           (board_player_two & COLUMN_0_FILLED == (board_player_two << SHIFT_6_COLUMNS) & COLUMN_0_FILLED) and \
           (board_player_two & COLUMN_1_FILLED == (board_player_two << SHIFT_4_COLUMNS) & COLUMN_1_FILLED) and \
           (board_player_two & COLUMN_2_FILLED == (board_player_two << SHIFT_2_COLUMNS) & COLUMN_2_FILLED)
