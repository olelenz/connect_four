from enum import Enum
from scipy import signal
import numpy as np
from typing import Callable, Optional, Tuple

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
    return 0b0000000_0000000_0000000_0000000_0000000_0000000_0000000, 0b0000000_0000000_0000000_0000000_0000000_0000000_0000000


def to_array(bitboard: int) -> np.ndarray:
    """
    Parameters
    ----------
    bitboard: int
        Bitboard to convert to an array.

    Returns
    -------
    :np.ndarray
        Bitboard as an array.
    """

    lst = str(bin(bitboard))[2:].rjust(49, '0')
    lst = np.reshape(list(lst), (7, 7))[::-1].T
    return np.array(lst, dtype=int)


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
    board = (np.add(to_array(board_player_one), to_array(board_player_two) * 2))[1::][::-1]
    output: str = "|==============|\n"
    for line in board[::-1]:
        output += "|"
        for entry in line:
            if entry == NO_PLAYER:
                output += NO_PLAYER_PRINT
            elif entry == PLAYER1:
                output += PLAYER1_PRINT
            else:
                output += PLAYER2_PRINT
            output += " "
        output += "|\n"
    output += "|==============|\n|0 1 2 3 4 5 6 |"
    return output


def string_to_board(pp_board: str) -> tuple[int, int]:  # TODO: change to binary
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

    if len(pp_board.split("\n")) != 9:  # using regex would be way better
        raise AttributeError
    output_player1 = ["0" for _ in range(49)]  # list("0000000_0000000_0000000_0000000_0000000_0000000_0000000")
    output_player2 = ["0" for _ in range(49)]  # list("0000000_0000000_0000000_0000000_0000000_0000000_0000000")
    for row, line in enumerate(pp_board.split("\n")[1:-2]):
        for column, entry in enumerate(line[1:-1:2]):
            if entry == PLAYER1_PRINT:
                output_player1[5 - row + 7 * column] = "1"  # 5 - row because string is bein looked at from the top
            elif entry == PLAYER2_PRINT:
                output_player2[5 - row + 7 * column] = "1"  # 7 * colum to jump columns
    output1 = "".join(output_player1[::-1])  # flip because binary is indexed from the first bit on the right
    output2 = "".join(output_player2[::-1])
    return int(output1, 2), int(output2, 2)


def apply_player_action(board_player_one: int, board_player_two: int, action: PlayerAction, player: BoardPiece) -> \
        tuple[int, int]:
    """
    Sets board[i, action] = player, where i is the lowest open row. Raises a ValueError
    if action is not a legal move. If it is a legal move, the modified version of the
    board is returned and the original board should remain unchanged (i.e., either set
    back or copied beforehand).

    Parameters
    ----------
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
    move_board = ((board_player_one | board_player_two) + (1 << action * 7))
    if move_board & (1 << (action * 7 + 6)):
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
    for i in [1, 6, 7, 8]:
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
    if player == PLAYER1:
        if connected_four(board_player_one):
            return GameState.IS_WIN
    else:
        if connected_four(board_player_two):
            return GameState.IS_WIN
    if board_player_one | board_player_two == 0b0111111_0111111_0111111_0111111_0111111_0111111_0111111:
        return GameState.IS_DRAW
    return GameState.STILL_PLAYING


def get_possible_moves(board_player_one: int, board_player_two: int, player: BoardPiece) -> [PlayerAction]:
    """
    Calculates all possible moves from a give board-position.

    Parameters
    ----------
    board_player_one: int
        Board PLAYER1.

    board_player_two: int
        Board PLAYER2.

    player : BoardPiece
        The player which wants to make a move.

    Returns
    -------
    :[PlayerAction]
        A list containing all possible moves.

    """
    if player == PLAYER2:  # no moves are possible if either player has already won
        if connected_four(board_player_one):
            return []
    else:
        if connected_four(board_player_two):
            return []
    board_full = board_player_one | board_player_two
    out: [PlayerAction] = [3, 2, 4, 1, 5, 0, 6]
    for i in [3, 2, 4, 1, 5, 0, 6]:
        if board_full & (1 << (i * 7 + 5)):
            out.remove(i)
    return out


def create_dictionary_key(board_player_one: int, board_player_two: int) -> int:
    """

    Parameters
    ----------
    board_player_one: int
        Board PLAYER1.

    board_player_two: int
        Board PLAYER2.

    Returns
    -------
    :int
        Key for the dictionary
    """
    return (board_player_one << 49) | board_player_two


def mirror_board(board_player1: int, board_player2: int) -> tuple[int, int]:
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
    Mirrors a single board string by bit shifting

    Parameters
    ----------
    player_board:
        The players board

    Returns
    -------
    int:
        The mirrored board
    """
    row1 = 0b0111111_0000000_0000000_0000000_0000000_0000000_0000000 & (player_board << 42)
    row2 = 0b0000000_0111111_0000000_0000000_0000000_0000000_0000000 & (player_board << 28)
    row3 = 0b0000000_0000000_0111111_0000000_0000000_0000000_0000000 & (player_board << 14)
    row4 = 0b0000000_0000000_0000000_0111111_0000000_0000000_0000000 & player_board
    row5 = 0b0000000_0000000_0000000_0000000_0111111_0000000_0000000 & (player_board >> 14)
    row6 = 0b0000000_0000000_0000000_0000000_0000000_0111111_0000000 & (player_board >> 28)
    row7 = 0b0000000_0000000_0000000_0000000_0000000_0000000_0111111 & (player_board >> 42)
    new_board = row1 | row2 | row3 | row4 | row5 | row6 | row7
    return new_board


def add_mirror_to_dictionary(board_player1: int, board_player2: int, dictionary: {}, alpha_beta: tuple[int, int]):
    """
    Uses the mirror functions to add a mirrored board, its evaluation and playeraction to the dictionary.

    Parameters
    ----------
    board_player1: int
        Board player1
    board_player2: int
        Board player2
    dictionary: {}
        Dictionary  # should be reference of dictionary
    alpha_beta
        Tuple that contains evaluation and playeraction

    """
    mirror_board_player1, mirror_board_player2 = mirror_board(board_player1, board_player2)
    mirror_key = create_dictionary_key(mirror_board_player1, mirror_board_player2)
    dictionary[mirror_key] = alpha_beta


def is_mirror_possible(board_player1: int, board_player2: int) -> bool:
    """
    Checks if the board could still have mirrored states in the future. E.g. if player1 has a piece in the bottom
    left corner and player2 in the bottom right corner, the board is asymmetric and mirrored board states
    will no longer occur.
    The goal is to not call mirror functions after this function returns False

    Parameters
    ----------
    board_player1: int
        Board player1
    board_player2: int
        Board player2

    Returns
    -------
    bool:
        if board can be mirrored of not
    """
    return (board_player1 & mirror_player_board(board_player2)) == 0
