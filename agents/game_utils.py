from enum import Enum
from scipy import signal

import numpy as np

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


def initialize_game_state() -> np.ndarray:
    """
    Returns an ndarray, shape (6, 7) and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER).
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
    """
    i: int = 0
    try:
        while board[i, action] != NO_PLAYER:
            i += 1
    except IndexError:
        raise ValueError
    output: np.ndarray = board.copy()
    output[i, action] = player
    return output


def connected_four(board: np.ndarray, player: BoardPiece) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.
    """
    relevant_pieces = initialize_game_state()
    for row in range(6):
        for column in range(7):
            if board[row, column] == player:
                relevant_pieces[5 - row, column] = 1  # notice the flipped rows
    convolve_2nds = [[[1, 1, 1, 1]], [[1], [1], [1], [1]], np.identity(4), np.flip(np.identity(4), 1)]
    for snd in convolve_2nds:
        if (signal.convolve2d(relevant_pieces, snd, mode="valid") == 4).any(): return True
    return False
    # explanation:
    # conv2d_horizontal = signal.convolve2d(relevant_pieces, [[1, 1, 1, 1]], mode="valid")
    # conv2d_vertical = signal.convolve2d(relevant_pieces, [[1], [1], [1], [1]], mode="valid")
    # conv2d_diagonal_left_top = signal.convolve2d(relevant_pieces, np.identity(4), mode="valid")
    # conv2d_diagonal_right_top = signal.convolve2d(relevant_pieces, np.flip(np.identity(4), 1), mode="valid")
    # return (conv2d_horizontal == 4).any() or (
    # conv2d_vertical == 4).any() or (conv2d_diagonal_left_top == 4).any() or (conv2d_diagonal_right_top == 4).any()


def check_end_state(board: np.ndarray, player: BoardPiece) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?
    """
    if connected_four(board, player): return GameState.IS_WIN
    if (board == 0).sum() == 0: return GameState.IS_DRAW #  TODO:  not only possibility for draw....
    return GameState.STILL_PLAYING
