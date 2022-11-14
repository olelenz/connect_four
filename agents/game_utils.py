from enum import Enum
from scipy import signal
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


class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0


GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]


def initialize_game_state() -> np.ndarray:
    """
    Returns an ndarray, shape (6, 7) and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER).

    Returns
    ----------
    numpy.ndarray
        Initial game state.
    """
    return np.zeros((6, 7), BoardPiece)


def pretty_print_board(board: np.ndarray) -> str:
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
    board: numpy.ndarray
        Board to be converted to a String.

    Returns
    ----------
    str
        String representation of the board.
    """
    output: str = "|==============|\n"
    for line in board[::-1]:
        output += "|"
        for entry in line:
            if entry == NO_PLAYER:
                output += NO_PLAYER_PRINT
            elif entry == PLAYER1:
                output += PLAYER1_PRINT
            elif entry == PLAYER2:
                output += PLAYER2_PRINT
            output += " "
        output += "|\n"
    output += "|==============|\n|0 1 2 3 4 5 6 |"
    return output


def string_to_board(pp_board: str) -> np.ndarray:
    """
    Takes the output of pretty_print_board and turns it back into an ndarray.
    This is quite useful for debugging, when the agent crashed and you have the last
    board state as a string.

    Parameters
    ----------
    pp_board: str
        String-representation of the board-position.

    Returns
    ----------
    numpy.ndarray
        The board generated from the input-String.
    """
    output: np.ndarray = initialize_game_state()
    for row, line in enumerate(pp_board.split("\n")[1:-2]):
        for column, entry in enumerate(line[1:-1:2]):
            if entry == PLAYER1_PRINT:
                output[5 - row, column] = PLAYER1
            elif entry == PLAYER2_PRINT:
                output[5 - row, column] = PLAYER2
    return output


def apply_player_action(board: np.ndarray, action: PlayerAction, player: BoardPiece) -> np.ndarray:
    """
    Sets board[i, action] = player, where i is the lowest open row. Raises a ValueError
    if action is not a legal move. If it is a legal move, the modified version of the
    board is returned and the original board should remain unchanged (i.e., either set
    back or copied beforehand).

    Parameters
    ----------
    board: numpy.ndarray
        The board-state to apply the action on.
    action: PlayerAction
        The action to be performed.
    player: BoardPiece
        The player which tries to make the desired move.

    Returns
    ----------
    numpy.ndarray
        Modified board-position if the move was legal.

    Raises
    ----------
    ValueError
        If the column in which the player wants to play is already filled.
    """
    i: int = 0
    try:
        while board[i, action] != NO_PLAYER:
            i += 1
    except IndexError:
        raise ValueError
    output: np.ndarray = board.copy()  # deep copy
    output[i, action] = player
    return output


def connected_four(board: np.ndarray, player: BoardPiece) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.

    Parameters
    ----------
    board: numpy.ndarray
        The board to check connected four on.
    player: BoardPiece
        Checks connected four for this player.

    Returns
    ----------
    :bool
        Returns True if there are four adjacent pieces equal to `player` arranged
        in either a horizontal, vertical, or diagonal line. Returns False otherwise.
    """
    relevant_pieces = initialize_game_state()
    relevant_pieces[board == player] = 1
    for snd in [[[1, 1, 1, 1]], [[1], [1], [1], [1]], np.identity(4), np.identity(4)[::-1, ::]]:
        if (signal.convolve2d(relevant_pieces, snd, mode="valid") == 4).any():
            return True
    return False


def check_end_state(board: np.ndarray, player: BoardPiece) -> GameState:
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


def get_possible_moves(board: np.ndarray) -> [PlayerAction]:
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
    for i in range(0, 7):
        if board[5][i] == NO_PLAYER:
            out.append(PlayerAction(i))
    return out

