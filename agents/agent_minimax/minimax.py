import copy
from typing import Tuple, Optional
from interruptingcow import timeout

from agents.agent_minimax.minimax_data import MinimaxCalculation
from agents.game_utils import *
from agents.saved_state import SavedState

FULL_BOARD: int = 0b0111111_0111111_0111111_0111111_0111111_0111111_0111111
START_VALUE: int = 100
MAX_VALUE: int = 1_000_000_000_000_000_000
MIN_MAX_FUNCTIONS = (min, max)


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
    dictio = {-1: {}}
    dictio_one = {-1: {}}
    dictio_two = {-1: {}}
    # use_mirror = is_mirror_possible(board_player_one, board_player_two)
    use_mirror = False
    minimax_data: MinimaxCalculation = MinimaxCalculation(depth=depth, board_player_one=board_player_one, board_player_two=board_player_two, current_player=player, dictionary=dictio)
    #print(minimax_data.__repr__())
    if player == PLAYER1:
        evaluation: list[int, [PlayerAction]] = \
            minimax_rec(minimax_data, alpha, beta, [], next_moves)  # start maximizing if PLAYER1 is to play
    else:
        minimax_data.__setattr__("minmax", 1)
        evaluation: list[int, [PlayerAction]] = \
            minimax_rec(depth, board_player_one, board_player_two, player, alpha,
                        beta, dictio_two, [], next_moves, 0, minimax_data)  # start minimizing if PLAYER2 is to play
    return evaluation


def generate_move_minimax(board_player_one: int, board_player_two: int, player: BoardPiece,
                          saved_state: Optional[SavedState], seconds: int = 4) -> Tuple[
    PlayerAction, Optional[SavedState]]:
    depth: int = 1
    evaluation: list[int, [PlayerAction]] = [0, []]
    move_output: int = -1
    try:
        with timeout(seconds, exception=RuntimeError):
            while True:
                evaluation: list[int, [PlayerAction]] = generate_move_minimax_id(board_player_one, board_player_two,
                                                                                 player, None, evaluation[1], depth)
                move_output = evaluation[1][0]
                print(depth, " moves: ", evaluation[1], " eval: ", evaluation[0])
                if depth >= len(evaluation[1]) + 4:
                    break
                depth += 1

    except RuntimeError:
        pass
    return PlayerAction(move_output), None


def minimax_rec(minimax_data: MinimaxCalculation, alpha: list[int, [PlayerAction]], beta: list[int, [PlayerAction]], next_moves: list[int], moves_line: list[int]) -> list[int, [PlayerAction]]:
    possible_moves, game_state = get_possible_moves_iterative(copy.deepcopy(minimax_data), next_moves=next_moves)
    if not possible_moves:
        return [handle_empty_moves_eval(copy.deepcopy(minimax_data), game_state), moves_line]
    if minimax_data.depth == 0:  # desired depth reached - recursion anchor
        evaluation: int = evaluate_board_using_windows(minimax_data.board_player_one, minimax_data.board_player_two)
        return [evaluation, moves_line]
    if minimax_data.minmax == 1:  # max
        for move in possible_moves:
            new_board_player_one, new_board_player_two = apply_player_action(
                (minimax_data.board_player_one, minimax_data.board_player_two, minimax_data.current_player), move)
            minimax_data.__setattr__("board_player_one", new_board_player_one)
            minimax_data.__setattr__("board_player_two", new_board_player_two)
            alpha = get_alpha(copy.deepcopy(minimax_data), move, alpha=alpha, beta=beta, moves_line=moves_line, next_moves=next_moves)
            if beta[0] <= alpha[0]:
                return alpha
        return alpha
    else:  # min
        for move in possible_moves:
            new_board_player_one, new_board_player_two = apply_player_action(
                (minimax_data.board_player_one, minimax_data.board_player_two, minimax_data.current_player), move)
            minimax_data.__setattr__("board_player_one", new_board_player_one)
            minimax_data.__setattr__("board_player_two", new_board_player_two)
            beta = get_beta(copy.deepcopy(minimax_data), move,alpha=alpha, beta=beta, moves_line=moves_line, next_moves=next_moves)
            if beta[0] <= alpha[0]:
                return beta
        return beta


def get_alpha(minimax_data: MinimaxCalculation, move: PlayerAction, alpha: list[int, [PlayerAction]], beta: list[int, [PlayerAction]], moves_line: list[int], next_moves: []):
    saved_eval = get_eval_from_dictionary(minimax_data.board_player_one, minimax_data.board_player_two, minimax_data.dictionary, moves_line)
    if saved_eval is not None:
        return MIN_MAX_FUNCTIONS[minimax_data.minmax]([alpha, saved_eval], key=lambda x: x[0])

    moves_line_new = moves_line.copy()
    moves_line_new.append(move)

    minimax_data.__setattr__("depth", minimax_data.depth-1)
    minimax_data.__setattr__("current_player", 3-minimax_data.current_player)
    minimax_data.__setattr__("minmax", 1 - minimax_data.minmax)
    recursion_eval = minimax_rec(copy.deepcopy(minimax_data), alpha=alpha, beta=beta, moves_line=moves_line, next_moves=next_moves)
    alpha = MIN_MAX_FUNCTIONS[minimax_data.minmax]([alpha, recursion_eval], key=lambda x: x[0])
    minimax_data.dictionary[minimax_data.board_player_one] = {minimax_data.board_player_two: [alpha[0], alpha[1][minimax_data.depth + 1:]]}  # possible mistake here
    return alpha


def get_beta(minimax_data: MinimaxCalculation, move: PlayerAction, alpha: list[int, [PlayerAction]], beta: list[int, [PlayerAction]], moves_line: [], next_moves: []):
    saved_eval = get_eval_from_dictionary(minimax_data.board_player_one, minimax_data.board_player_two, minimax_data.dictionary, moves_line)
    if saved_eval is not None:
        return MIN_MAX_FUNCTIONS[minimax_data.minmax]([beta, saved_eval], key=lambda x: x[0])

    moves_line_new = moves_line.copy()
    moves_line_new.append(move)
    minimax_data.__setattr__("depth", minimax_data.depth - 1)
    minimax_data.__setattr__("current_player", 3 - minimax_data.current_player)
    minimax_data.__setattr__("minmax", 1 - minimax_data.minmax)
    recursion_eval = minimax_rec(copy.deepcopy(minimax_data), alpha=alpha, beta=beta, moves_line=moves_line, next_moves=next_moves)
    beta = MIN_MAX_FUNCTIONS[minimax_data.minmax]([beta, recursion_eval], key=lambda x: x[0])
    minimax_data.dictionary[minimax_data.board_player_one] = {minimax_data.board_player_two: [beta[0], beta[1][minimax_data.depth + 1:]]}  # possible mistake here
    return beta


def get_eval_from_dictionary(new_board_player_one: int, new_board_player_two: int, dictionary: {}, moves_line: [PlayerAction]) -> int | None:
    try:
        saved_eval = dictionary[new_board_player_one][new_board_player_two]
        saved_eval[1] = moves_line + saved_eval[1]
        return saved_eval
    except KeyError:
        return None


def get_possible_moves_iterative(minimax_data: MinimaxCalculation, next_moves: []) -> ([PlayerAction], GameState):
    try:
        return get_possible_moves(minimax_data.board_player_one, minimax_data.board_player_two, minimax_data.current_player, next_moves.pop(0))
    except IndexError:
        return get_possible_moves(minimax_data.board_player_one, minimax_data.board_player_two, minimax_data.current_player)


def handle_empty_moves_eval(minimax_data: MinimaxCalculation, game_state: GameState) -> int:
    if game_state == GameState.IS_WIN:
        if minimax_data.current_player == PLAYER1:
            return int(-START_VALUE * 2 ** (minimax_data.depth))
        else:
            return int(START_VALUE * 2 ** (minimax_data.depth))
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


def number_of_possible_4_connected_left(board_player1: int, board_player2: int) -> int:
    empty_positions = empty_board_positions(board_player1, board_player2)
    player1_possible_4connected = number_of_connected_n(empty_positions & board_player1, 4)
    player2_possible_4connected = number_of_connected_n(empty_positions & board_player2, 4)
    return player1_possible_4connected - player2_possible_4connected


def empty_board_positions(board_player1: int, board_player2: int) -> int:
    return 0b0111111_0111111_0111111_0111111_0111111_0111111_0111111 - board_player1 - board_player2


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
        return 10
    elif counter_player2 == 3:
        return -10
    return counter_player1 - counter_player2


def list_windows() -> [(int, int, int, int)]:
    """
    Builds windows that are represented as board with a single piece (1) by shifting the number 1 in different amounts.

    Returns
    -------
    [int]:
        List of 69 possible 4-in-a-row windows as tuples.
        24 horizontal, 21 vertical, 12 diagonal-up, 12 diagonal-down
    """
    # 0b0000000_0000000_0000000_0000000_0000000_0000000_0000001 for reference
    result: [(int, int, int, int)] = []

    # horizontal windows
    for column_offset in range(4):
        for row_offset in range(6):
            result += [(1 << (47 - 7 * column_offset - row_offset), 1 << (40 - 7 * column_offset - row_offset),
                        1 << (33 - 7 * column_offset - row_offset), 1 << (26 - 7 * column_offset - row_offset))]

    # vertical windows
    for column_offset in range(7):
        for row_offset in range(3):
            result += [(1 << (47 - 7 * column_offset - row_offset), 1 << (46 - 7 * column_offset - row_offset),
                        1 << (45 - 7 * column_offset - row_offset), 1 << (44 - 7 * column_offset - row_offset))]

    # diagonal-up windows
    for position in [47, 46, 45, 40, 39, 38, 33, 32, 31, 26, 25, 24]:
        result += [(1 << position, 1 << (position - 8), 1 << (position - 16), 1 << (position - 24))]

    # diagonal-down window
    for position in [42, 43, 44, 35, 36, 37, 28, 29, 30, 21, 22, 23]:
        result += [(1 << position, 1 << (position - 6), 1 << (position - 12), 1 << (position - 18))]

    return result


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
    for window in list_windows():
        board_score += evaluate_window(window, board_player1, board_player2)
    return board_score

