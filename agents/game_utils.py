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
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
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
    board = (np.add(to_array(board_player_one), to_array(board_player_two)*2))[1::][::-1]
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


def string_to_board(pp_board: str) -> np.ndarray:  # TODO: change to binary
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
    numpy.ndarray
        The board generated from the input-String.
    """

    output: np.ndarray = initialize_game_state()
    if len(pp_board.split("\n")) != 9:  # using regex would be way better
        raise AttributeError
    for row, line in enumerate(pp_board.split("\n")[1:-2]):
        for column, entry in enumerate(line[1:-1:2]):
            if entry == PLAYER1_PRINT:
                output[5 - row, column] = PLAYER1
            elif entry == PLAYER2_PRINT:
                output[5 - row, column] = PLAYER2
    return output


def apply_player_action(board_player_one: int, board_player_two: int, action: PlayerAction, player: BoardPiece) -> tuple[int, int]:
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
    if move_board & (1 << (action*7 + 6)):
        raise ValueError
    if player == PLAYER1:
        return (move_board - board_player_two) | board_player_one, board_player_two
    else:
        return board_player_one, (move_board - board_player_one) | board_player_two


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
        if temp_bitboard & (temp_bitboard >> 2*i):
            return True
    return False



def check_end_state(board: np.ndarray, player: BoardPiece) -> GameState:  # TODO: change to binary
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?

    Parameters
    ----------
    board: numpy.ndarray
        The board to check the end state on.
    player: BoardPiece
        Checks the end-state for this player.

    Returns
    ----------
    :GameState
        The current game-state - either IS_WIN, IS_DRAW or STILL_PLAYING.
    """

    if connected_four(board, player):
        return GameState.IS_WIN
    if (board == 0).sum() == 0:
        return GameState.IS_DRAW
    return GameState.STILL_PLAYING


def get_possible_moves(board: np.ndarray) -> [PlayerAction]:  # TODO: change to binary
    """
    Calculates all possible moves from a give board-position.

    Parameters
    ----------
    board: numpy.ndarray
        The moves are calculated from this board-position.

    Returns
    -------
    :[PlayerAction]
        A list containing all possible moves.

    """
    out: [PlayerAction] = []
    if connected_four(board, PLAYER1) or connected_four(board, PLAYER2):  # no moves are possible if either player
        # has already won
        return []
    for i in [3, 2, 4, 1, 5, 0, 6]:  # standard: range(0,7), better moves in the middle -> check first: [3, 2, 4, 1, 5, 0, 6]
        # minimax from 2.475s to 0.22s (depth 6) and from 51.605s to 18.963s (depth 7) - first move
        if board[5][i] == NO_PLAYER:
            out.append(PlayerAction(i))
    return out
