from enum import Enum
import numpy as np
from typing import Callable, Optional
from agents.saved_state import SavedState

BoardPiece = np.int8  # The data type (dtype) of the board
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 (player to move first) has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 (player to move second) has a piece

BoardPiecePrint = str  # dtype for string representation of BoardPiece
NO_PLAYER_PRINT = BoardPiecePrint(' ')
PLAYER1_PRINT = BoardPiecePrint('X')
PLAYER2_PRINT = BoardPiecePrint('O')

PlayerAction = np.int8  # The column to be played

# static variables
EMPTY_BOARD: int = 0b0000000_0000000_0000000_0000000_0000000_0000000_0000000
FULL_BOARD: int = 0b0111111_0111111_0111111_0111111_0111111_0111111_0111111
BOARD_SHAPE_BINARY: (int, int) = (7, 7)
BINARY_SIZE: int = 49
PRINT_SUBSTITUTION_TABLE: {int: str} = {0: ' ', 1: 'X', 2: 'O'}
PRINT_BACK_SUBSTITUTION_TABLE_PLAYER_ONE: {str: int} = {' ': 0, 'X': 1, 'O': 0}
PRINT_BACK_SUBSTITUTION_TABLE_PLAYER_TWO: {str: int} = {' ': 0, 'X': 0, 'O': 1}
TOP_PRETTY_PRINT_BOARD: str = "|==============|\n"
BOTTOM_PRETTY_PRINT_BOARD: str = "|==============|\n|0 1 2 3 4 5 6 |"
SIDE_PRETTY_PRINT_BOARD: str = "|"
EMPTY_ROW_CHAR: [str] = [' ', ' ', ' ', ' ', ' ', ' ', ' ']
HEIGHT_PRINT_BOARD: int = 9
HEIGHT_TOP_PRINT_BOARD: int = 1
HEIGHT_BOTTOM_PRINT_BOARD: int = 2
HEIGHT_BOARD: int = HEIGHT_PRINT_BOARD - HEIGHT_TOP_PRINT_BOARD - HEIGHT_BOTTOM_PRINT_BOARD
MOVE_ORDER: [PlayerAction] = [3, 2, 4, 1, 5, 0, 6]
DICT_EMPTY_BOARDS: {int: {int: [int, [int]]}} = {EMPTY_BOARD: {EMPTY_BOARD: [0, [3, 3, 3, 3, 3]]}}


class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0


GenMove = Callable[
    [int, int, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]


def initialize_game_state() -> tuple[int, int]:
    """
    Returns a tuple containing a number for PLAYER1 and a second number for PLAYER2 , shape (7, 7), initialized to 0 (NO_PLAYER).

    Returns
    ----------
    tuple[int, int]
        Initial game state.
    """
    return EMPTY_BOARD, EMPTY_BOARD


def to_array(bitboard: int) -> np.ndarray:
    """
    Function to convert a bitboard to a numpy nd-array for visualisation.

    Parameters
    ----------
    bitboard: int
        Bitboard to convert to an array.

    Returns
    -------
    :np.ndarray
        Bitboard as an array.
    """
    bitboard_string = str(bin(bitboard))[2:].rjust(BINARY_SIZE, '0')  # to String, remove 0b and pad 0 to the left
    assert len(bitboard_string) == BINARY_SIZE  # catch too long Strings
    return np.rot90(np.fromiter(bitboard_string, dtype=int).reshape(BOARD_SHAPE_BINARY), 3)  # read to numpy array and reshape to fit board


def pretty_print_board(board_player_one: int, board_player_two: int) -> str:
    """
    Should return `board` converted to a human readable string representation,
    to be used when playing or printing diagnostics to the console (stdout). The piece in
    board[0, 0] should appear in the lower-left. Here's an example output, note that we use
    PLAYER1_Print to represent PLAYER1 and PLAYER2_Print to represent PLAYER2):
    |==============|
    |              |
    |              |
    |    X X       |
    |    O X X     |
    |  O X O O     |
    |  O O X X     |
    |==============|
    |0 1 2 3 4 5 6 |

    Parameters
    ----------
    board_player_one: int
        Board PLAYER1 to be converted to a String.

    board_player_two: int
        Board PLAYER2 to be converted to a String.

    Returns
    ----------
    str
        String representation of the board.
    """
    board: np.ndarray = np.add(to_array(board_player_one), to_array(board_player_two) * 2)

    replace_board: Callable = np.vectorize(lambda dictionary, key: dictionary.get(key))
    board_modified: np.ndarray = replace_board(PRINT_SUBSTITUTION_TABLE, board)

    body: str = ""
    for index in range(1, HEIGHT_BOARD + 1):
        body += SIDE_PRETTY_PRINT_BOARD+''.join(map('{} '.format, board_modified[index]))+SIDE_PRETTY_PRINT_BOARD+"\n"

    return TOP_PRETTY_PRINT_BOARD + body + BOTTOM_PRETTY_PRINT_BOARD


def string_to_board(pp_board: str) -> tuple[int, int]:
    """
    Takes the output of pretty_print_board and turns it back into an ndarray.
    This is quite useful for debugging, when the agent crashed and you have the last
    board state as a string.

    Parameters
    ----------
    pp_board: str
        String-representation of the board-position.

    Raises
    ----------
    AttributeError
        If the input string is not representing a valid board state.

    Returns
    ----------
    :tuple[int, int]
        Board-positions as binary numbers.
    """
    if len(pp_board.split("\n")) != HEIGHT_PRINT_BOARD:
        raise AttributeError
    string_input: str = pp_board.replace(SIDE_PRETTY_PRINT_BOARD, '')
    string_input: list[str] = string_input.split("\n")[HEIGHT_TOP_PRINT_BOARD:-HEIGHT_BOTTOM_PRINT_BOARD]
    get_entries: Callable = np.vectorize(lambda arr: list(arr[::2]))
    entry_array = list(map(get_entries, string_input))
    entry_array = np.vstack((EMPTY_ROW_CHAR, entry_array))  # add buffer
    entry_array = np.rot90(entry_array, 1)
    replace_board: Callable = np.vectorize(lambda dictionary, key: dictionary.get(key))
    entry_array_numbers_player_one: np.ndarray = replace_board(PRINT_BACK_SUBSTITUTION_TABLE_PLAYER_ONE, entry_array).flatten()
    output_player_one: int = int(np.array2string(entry_array_numbers_player_one, separator='')[1:-1], 2)
    entry_array_numbers_player_two: np.ndarray = replace_board(PRINT_BACK_SUBSTITUTION_TABLE_PLAYER_TWO, entry_array).flatten()
    output_player_two: int = int(np.array2string(entry_array_numbers_player_two, separator='')[1:-1], 2)

    return output_player_one, output_player_two


def apply_player_action(board_information: (int, int, BoardPiece), action: PlayerAction) -> \
        tuple[int, int]:
    """
    Sets board[i, action] = player, where i is the lowest open row. Raises a ValueError
    if action is not a legal move. If it is a legal move, the modified version of the
    board is returned and the original board should remain unchanged (i.e., either set
    back or copied beforehand).

    Parameters
    ----------
    board_information: (int, int, BoardPiece)
        Contains the board for PLAYER1, PLAYER2 and the next player to make a move.

    board_player_one: int
        Board PLAYER1.

    board_player_two: int
        Board PLAYER2.

    action: PlayerAction
        The action to be performed.

    player: BoardPiece
        The player which tries to make the desired move.

    Raises
    ----------
    ValueError
        If the column in which the player wants to play is already filled.

    Returns
    ----------
    :tuple[int, int]
        Modified board-positions if the move was legal.
    """
    board_player_one: int
    board_player_two: int
    player: BoardPiece
    board_player_one, board_player_two, player = board_information
    move_board = ((board_player_one | board_player_two) + (1 << action * (HEIGHT_BOARD + 1)))
    if move_board & (1 << (action * (HEIGHT_BOARD + 1) + HEIGHT_BOARD)):
        raise ValueError
    if player == PLAYER1:
        return (move_board & ~board_player_two) | board_player_one, board_player_two
    else:
        return board_player_one, (move_board & ~board_player_one) | board_player_two


def connected_four(board: int) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.

    Parameters
    ----------
    board: int
        Board for the player to check connected four on.
    Returns
    ----------
    :bool
        Returns True if there are four adjacent pieces equal to `player` arranged
        in either a horizontal, vertical, or diagonal line. Returns False otherwise.
    """
    for i in [1, HEIGHT_BOARD, HEIGHT_BOARD+1, HEIGHT_BOARD+2]:
        temp_bitboard = board & (board >> i)
        if temp_bitboard & (temp_bitboard >> 2 * i):
            return True
    return False


def check_end_state(board_player_one: int, board_player_two: int, player: BoardPiece) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?

    Parameters
    ----------
    board_player_one: int
        Board PLAYER1.

    board_player_two: int
        Board PLAYER2.

    player: BoardPiece
        The player to check the state.

    Returns
    ----------
    :GameState
        The current game-state - either IS_WIN, IS_DRAW or STILL_PLAYING.
    """
    if player == PLAYER1 and connected_four(board_player_one):
        return GameState.IS_WIN
    elif connected_four(board_player_two):
        return GameState.IS_WIN
    if board_player_one | board_player_two == FULL_BOARD:
        return GameState.IS_DRAW
    return GameState.STILL_PLAYING


def get_possible_moves(board_player_one: int, board_player_two: int, player: BoardPiece, next_move: int = 3) -> ([PlayerAction], GameState):
    """
    Calculates all possible moves from a give board-position.

    Parameters
    ----------
    board_player_one: int
        Board PLAYER1.

    board_player_two: int
        Board PLAYER2.

    player: BoardPiece
        The player which wants to make a move.

    next_move:
        The move which should be first in returned list if it is allowed. Used for better pruning in iterative deepening.

    Returns
    -------
    :([PlayerAction], GameState)
        A list containing all possible moves and the current GameState.
    """
    game_state: GameState = check_end_state(board_player_one, board_player_two, 3-player)  # win needs to be checked on the other players board
    if game_state == GameState.IS_WIN:  # no moves are possible if either player has already won
        return [], game_state
    board_both_players = board_player_one | board_player_two
    out: [PlayerAction] = MOVE_ORDER.copy()
    out.insert(0, out.pop(out.index(next_move)))  # move preferred move to front
    for i in MOVE_ORDER:
        if board_both_players & (1 << (i * (HEIGHT_BOARD+1) + (HEIGHT_BOARD-1))):  # check for top piece in column
            out.remove(i)
    return out, game_state


