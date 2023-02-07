from typing import Tuple

from interruptingcow import timeout

from agents.game_utils import *

# static variables:
ALPHA: [int, [PlayerAction]] = [-1_000_000_000_000_000_000, [PlayerAction(-8)]]
BETA: [int, [PlayerAction]] = [1_000_000_000_000_000_000, [PlayerAction(-7)]]
INIT_DICTIONARY: {int, dict} = {-1: {}}


def generate_move_minimax(board_player_one: int, board_player_two: int, player: BoardPiece,
                          saved_state: Optional[SavedState], seconds: int = 4) -> Tuple[
    PlayerAction, Optional[SavedState]]:
    depth: int = 0
    evaluation: list[int, [PlayerAction]] = [0, []]
    board_information: (int, int, BoardPiece) = (board_player_one, board_player_two, player)
    try:
        with timeout(seconds, exception=RuntimeError):
            while True:
                evaluation: list[int, [PlayerAction]] = generate_move_minimax_start_single_depth(board_information, evaluation[1], depth)
                print(depth, " moves: ", evaluation[1], " eval: ", evaluation[0])
                if depth >= len(evaluation[1]) + 4:
                    break
                depth += 1

    except RuntimeError:
        pass
    return PlayerAction(evaluation[1][0]), None


def generate_move_minimax_start_single_depth(board_information: (int, int, BoardPiece), next_moves: list[int], depth: int = 8) -> list[int, [PlayerAction]]:
    """
    Generates the next move using the minimax algorithm.
    Parameters
    ----------
    board_information: (int, int, BoardPiece)
        Contains the board for PLAYER1, PLAYER2 and the next player to make a move.
    next_moves: list[int]
        Predicted line of moves to be used for more efficient pruning.
    depth: int
        Depth of the search tree to stop calculating.
    Returns
    -------
    :list[int, [PlayerAction]]
        List containing the evaluation and the predicted line of moves.
    """
    if board_information[2] == PLAYER1:
        evaluation: list[int, [PlayerAction]] = minimax_rec(board_information, [next_moves, []], (depth, 0),
                                                            INIT_DICTIONARY.copy(), [ALPHA.copy(), BETA.copy()])
    else:
        evaluation: list[int, [PlayerAction]] = minimax_rec(board_information, [next_moves, []], (depth, 1),
                                                            INIT_DICTIONARY.copy(), [ALPHA.copy(), BETA.copy()])
    return evaluation


def minimax_rec(board_information: (int, int, BoardPiece), moves: [list[int], list[int]], rec_info: (int, int), dictionary: {}, alpha_beta: [list[int, [PlayerAction]], list[int, [PlayerAction]]]) -> (int, [PlayerAction]):  # moves: next and line, rec_info: 0 is depth, 1 is flag for alpha/beta
    possible_moves: [int]
    game_state: GameState
    possible_moves, game_state = get_possible_moves_iterative(board_information, moves[0])
    if not possible_moves:
        return handle_empty_moves_eval(board_information[2], game_state, rec_info[0]), moves[1]
    if rec_info[0] == 0:
        evaluation: int = evaluate_board_using_windows(board_information[0], board_information[1])  # TODO: add correct method
        return evaluation, moves[1]
    for move in possible_moves:
        new_board_player_one, new_board_player_two = apply_player_action(board_information, move)
        new_board_information: (int, int, BoardPiece) = (new_board_player_one, new_board_player_two, BoardPiece(3 - board_information[2]))
        get_alpha_beta(new_board_information, moves, rec_info, dictionary, alpha_beta, move)  # modifies alpha_beta and dictionary
        if alpha_beta[1][0] <= alpha_beta[0][0]:
            return alpha_beta[rec_info[1]]
    return alpha_beta[rec_info[1]]


def get_alpha_beta(new_board_information: (int, int, BoardPiece), moves: [list[int], list[int]], rec_info: (int, int), dictionary: {}, alpha_beta: [list[int, [PlayerAction]], list[int, [PlayerAction]]], move: PlayerAction):
    min_max = max
    if rec_info[1] == 1:  # 0 is alpha,max --- 1 is beta,min
        min_max = min
    try:
        saved_eval = dictionary[new_board_information[0]][new_board_information[1]]
        saved_eval[1] = moves[1] + saved_eval[1]
        alpha_beta[rec_info[1]] = min_max([alpha_beta[rec_info[1]], saved_eval], key=lambda x: x[0])
    except KeyError:
        moves_line_new = moves[1].copy()
        moves_line_new.append(move)
        recursion_eval = minimax_rec(new_board_information, [moves[0], moves_line_new], (rec_info[0] - 1, 1 - rec_info[1]), dictionary, alpha_beta)
        alpha_beta[rec_info[1]] = min_max([alpha_beta[rec_info[1]], recursion_eval], key=lambda x: x[0])
        dictionary[new_board_information[0]] = {new_board_information[1]: [alpha_beta[rec_info[1]][0], alpha_beta[rec_info[1]][1][-rec_info[0]-1:]]}  # possible mistake here -> -depth+1


def get_possible_moves_iterative(board_information: (int, int, BoardPiece), next_moves: list[int]) -> ([PlayerAction], GameState):
    try:
        return get_possible_moves(board_information[0], board_information[1], board_information[2], next_moves.pop(0))
    except IndexError:
        return get_possible_moves(board_information[0], board_information[1], board_information[2])


def handle_empty_moves_eval(player: BoardPiece, game_state: GameState, depth: int) -> int:
    if game_state == GameState.IS_WIN:
        if player == PLAYER1:
            return int(-1_000_000_000_000_000_000 * 2 ** (depth))
        else:
            return int(1_000_000_000_000_000_000 * 2 ** (depth))
    elif game_state == GameState.IS_DRAW:
        return 0
    else:
        raise AttributeError

































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
            result += [(1 << (47-7*column_offset-row_offset), 1 << (40-7*column_offset-row_offset),
                       1 << (33-7*column_offset-row_offset), 1 << (26-7*column_offset-row_offset))]

    # vertical windows
    for column_offset in range(7):
        for row_offset in range(3):
            result += [(1 << (47-7*column_offset-row_offset), 1 << (46-7*column_offset-row_offset),
                       1 << (45-7*column_offset-row_offset), 1 << (44-7*column_offset-row_offset))]

    # diagonal-up windows
    for position in [47, 46, 45, 40, 39, 38, 33, 32, 31, 26, 25, 24]:
        result += [(1 << position, 1 << (position-8), 1 << (position-16), 1 << (position-24))]

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

