import numpy as np
import pytest
from agents.game_utils import *

EMPTY_BOARD: int = 0b0000000_0000000_0000000_0000000_0000000_0000000_0000000
EMPTY_BOARD_STRING: str = "|==============|\n|              |\n|              |\n|              |\n|              |\n|              |\n|              |\n|==============|\n|0 1 2 3 4 5 6 |"
FIRST_PIECE_BOARD: int = 0b0000000_0000000_0000000_0000001_0000000_0000000_0000000
FIRST_PIECE_STRING: str = "|==============|\n|              |\n|              |\n|              |\n|              |\n|              |\n|      X       |\n|==============|\n|0 1 2 3 4 5 6 |"
LEFT_BOTTOM_CORNER_PIECE_BOARD: int = 0b0000000_0000000_0000000_0000000_0000000_0000000_0000001
LEFT_BOTTOM_CORNER_PIECE_STRING: str = "|==============|\n|              |\n|              |\n|              |\n|              |\n|              |\n|X             |\n|==============|\n|0 1 2 3 4 5 6 |"
FULL_BOARD: int = 0b0111111_0111111_0111111_0111111_0111111_0111111_0111111
DIAGONAL_BOARD_RIGHT_TOP: int = 0b0000000_0000000_0000000_0001000_0000100_0000010_0000001
DIAGONAL_BOARD_RIGHT_TOP_PLAYER_ONE_STRING: str = "|==============|\n|              |\n|              |\n|      X       |\n|    X         |\n|  X           |\n|X             |\n|==============|\n|0 1 2 3 4 5 6 |"
DIAGONAL_BOARD_RIGHT_TOP_PLAYER_TWO_STRING: str = "|==============|\n|              |\n|              |\n|      O       |\n|    O         |\n|  O           |\n|O             |\n|==============|\n|0 1 2 3 4 5 6 |"
BOARD_SHAPE_BINARY = (7, 7)
PRINT_SUBSTITUTION_TABLE = {0: ' ', 1: 'X', 2: 'O'}
PRINT_BACK_SUBSTITUTION_TABLE_PLAYER_ONE = {' ': 0, 'X': 1, 'O': 0}
PRINT_BACK_SUBSTITUTION_TABLE_PLAYER_TWO = {' ': 0, 'X': 0, 'O': 1}
EMPTY_ROW_CHAR = [' ', ' ', ' ', ' ', ' ', ' ', ' ']
def test_initialize_game_state():
    board_player_one, board_player_two = initialize_game_state()
    assert board_player_one == EMPTY_BOARD
    assert board_player_two == EMPTY_BOARD


def test_to_array():
    ret = to_array(DIAGONAL_BOARD_RIGHT_TOP)
    array_board = np.zeros(BOARD_SHAPE_BINARY)
    array_board[6][0] = 1
    array_board[5][1] = 1
    array_board[4][2] = 1
    array_board[3][3] = 1
    assert np.array_equal(array_board, ret)


def test_to_array_assertion_error():
    with pytest.raises(AssertionError):
        to_array(FULL_BOARD << 2)


def test_pretty_print_board_empty():
    ret = pretty_print_board(EMPTY_BOARD, EMPTY_BOARD)
    assert isinstance(ret, str)
    assert ret == EMPTY_BOARD_STRING


def test_pretty_print_board_diagonal_right_top():
    ret = pretty_print_board(DIAGONAL_BOARD_RIGHT_TOP, EMPTY_BOARD)
    assert isinstance(ret, str)
    assert ret == DIAGONAL_BOARD_RIGHT_TOP_PLAYER_ONE_STRING


def test_pretty_print_board_middle_one():
    ret = pretty_print_board(FIRST_PIECE_BOARD, EMPTY_BOARD)
    assert isinstance(ret, str)
    assert ret == FIRST_PIECE_STRING


def test_string_to_board_empty():
    ret = string_to_board(EMPTY_BOARD_STRING)
    assert type(ret[0]) == int
    assert type(ret[1]) == int
    assert ret[0] == EMPTY_BOARD
    assert ret[1] == EMPTY_BOARD


def test_string_to_board_left_bottom_corner():
    ret = string_to_board(LEFT_BOTTOM_CORNER_PIECE_STRING)
    assert ret[0] == LEFT_BOTTOM_CORNER_PIECE_BOARD
    assert ret[1] == EMPTY_BOARD


def test_string_to_board_diagonal_board_right_top_player_one():
    ret = string_to_board(DIAGONAL_BOARD_RIGHT_TOP_PLAYER_ONE_STRING)
    assert ret[0] == DIAGONAL_BOARD_RIGHT_TOP
    assert ret[1] == EMPTY_BOARD


def test_string_to_board_diagonal_board_right_top_player_two():
    ret = string_to_board(DIAGONAL_BOARD_RIGHT_TOP_PLAYER_TWO_STRING)
    assert ret[0] == EMPTY_BOARD
    assert ret[1] == DIAGONAL_BOARD_RIGHT_TOP


def test_new_idea():
    pass


