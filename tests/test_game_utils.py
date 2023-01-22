import pytest

from agents.game_utils import *
import numpy as np
from collections import defaultdict


def test_run_all():
    test_initialize_game_state()
    test_pretty_print_board()
    test_string_to_board()
    test_apply_player_action()
    test_connected_four_general()
    test_connected_four_vertical()
    test_connected_four_horizontal()
    test_connected_four_diagonal_one()
    test_connected_four_diagonal_two()
    test_check_end_state()
    test_get_possible_moves()


def test_initialize_game_state():
    ret = initialize_game_state()

    assert isinstance(ret[0], int)
    assert isinstance(ret[1], int)
    assert np.all(ret[0] == 0)
    assert np.all(ret[1] == 0)


def test_pretty_print_board():
    board_p1, board_p2 = initialize_game_state()

    board_p1 = board_p1 | 0b1
    board_p2 = board_p2 | 0b1_0000000

    ret = pretty_print_board(board_p1, board_p2)

    assert isinstance(ret, str)
    assert ret == "|==============|\n|              |\n|              |\n|              |\n|              |\n|        " \
                  "      |\n|X O           |\n|==============|\n|0 1 2 3 4 5 6 |"


def test_string_to_board():
    board = 0b0000000_0000000_0000000_0000001_0000000_0000000_0000000, \
            0b0000000_0000000_0000000_0000000_0000001_0000000_0000000
    ret = string_to_board(
        "|==============|\n|              |\n|              |\n|              |\n|              |\n|        "
        "      |\n|    O X       |\n|==============|\n|0 1 2 3 4 5 6 |")
    print(bin(ret[0]))
    assert isinstance(ret[0], int)
    assert ret == board
    with pytest.raises(AttributeError):
        string_to_board("wrong format")


def test_apply_player_action():
    board_p1 = 0b0000000_0000000_0000000_0010101_0000000_0000000_0000000
    board_p2 = 0b0000000_0000000_0000000_0001010_0000000_0000000_0000000

    ret = apply_player_action(board_p1, board_p2, PlayerAction(3), PLAYER2)
    assert isinstance(ret[0], int)
    assert ret[1] != board_p2
    assert ret[1] == 0b0000000_0000000_0000000_0101010_0000000_0000000_0000000

    ret = apply_player_action(board_p1, board_p2, PlayerAction(2), PLAYER1)
    assert ret[0] == 0b0000000_0000000_0000000_0010101_0000001_0000000_0000000

    board_p2 = 0b0000000_0000000_0000000_0101010_0000000_0000000_0000000
    with pytest.raises(ValueError):
        apply_player_action(board_p1, board_p2, PlayerAction(3), PLAYER1)


def test_connected_four_general():
    ret = connected_four(0b0000000_0000000_0000000_00000000_0000000_0000000_0000000)
    assert not ret


def test_connected_four_vertical():
    # vertical
    ret = connected_four(0b0000000_0000000_0000000_0000000_0000000_0000000_0001111)
    assert ret


def test_connected_four_horizontal():
    # horizontal
    ret = connected_four(0b0000000_0000000_0000000_0000001_0000001_0000001_0000001)
    assert ret


def test_connected_four_diagonal_one():
    # diagonal left top
    ret = connected_four(0b0000000_0000000_0000000_0000001_0000010_0000100_0001000)
    assert ret


def test_connected_four_diagonal_two():
    # diagonal right top
    ret = connected_four(0b0000000_0000000_0000000_0001000_0000100_0000010_0000001)
    assert ret


def test_check_end_state():
    board_p1 = 0b0000000_0000000_0000000_0000000_0000001_0000000_0000000
    board_p2 = 0b0000000_0000000_0000000_0000000_0000000_0000001_0000000

    ret = check_end_state(board_p1, board_p2, PLAYER1)

    assert isinstance(ret, GameState)
    assert ret == GameState.STILL_PLAYING

    board_p1 = 0b0000000_0000000_0000000_0000000_0001111_0000000_0000000

    ret = check_end_state(board_p1, board_p2, PLAYER1)

    assert isinstance(ret, GameState)
    assert ret == GameState.IS_WIN

    board_p1 = 0b0101011_0101011_0111000_0010101_0010100_0100010_0101001
    board_p2 = 0b0010100_0010100_0000111_0101010_0101011_0011101_0010110

    ret = check_end_state(board_p1, board_p2, PLAYER1)

    assert isinstance(ret, GameState)
    assert ret == GameState.IS_DRAW


def test_check_end_state_still_playing():
    board_p1 = 0b0000000_0000000_0000000_0011101_0000011_0000000_0000000
    board_p2 = 0b0000001_0000000_0000011_0100010_0000100_0000000_0000000
    ret = check_end_state(board_p1, board_p2, PLAYER1)

    assert isinstance(ret, GameState)
    assert ret == GameState.STILL_PLAYING


def test_check_end_state_draw():
    board_p1 = 0b0001010_0010101_0111011_0101110_0000100_0010100_0100011
    board_p2 = 0b0110101_0101010_0000100_0010001_0111011_0101011_0011100
    ret = check_end_state(board_p1, board_p2, PLAYER1)

    assert isinstance(ret, GameState)
    assert ret == GameState.IS_DRAW


def test_get_possible_moves():
    board_p1 = 0b0100000_0000000_0100000_0000000_0100000_0000000_0100000
    board_p2 = 0b0000000_0100000_0000000_0100000_0000000_0100000_0000000
    ret = get_possible_moves(board_p1, board_p2, PLAYER1)

    assert ret == []

    board_p1 = 0b0000000_0000000_0000000_0000000_0000000_0000000_0000000
    board_p2 = 0b0000000_0000000_0000000_0000000_0000000_0000000_0000000

    ret = get_possible_moves(board_p1, board_p2, PLAYER1)
    assert ret == [3, 2, 4, 1, 5, 0, 6]

    board_p1 = 0b0000000_0000000_0000000_0000111_0000000_0000000_0000000
    board_p2 = 0b0000000_0000000_0000000_0111000_0000000_0000000_0000000

    ret = get_possible_moves(board_p1, board_p2, PLAYER1)

    assert ret == [2, 4, 1, 5, 0, 6]

    board_p1 = 0b0000000_0000000_0000000_0000111_0001111_0000000_0000000

    ret = get_possible_moves(board_p1, board_p2, PLAYER2)
    assert ret == []


def test_mirror_board():
    board1 = 0b0000000_0111111_0001000_0000000_0000000_0000000_0111111
    board2 = 0b0111111_0000000_0000000_0000000_0001000_0111111_0000000
    assert mirror_board(board1, board2) == (board2, board1)


def test_mirror_player_board():
    board1 = 0b0000000_0111111_0000000_0100000_0111111_0000000_0111111
    board2 = 0b0111111_0000000_0111111_0100000_0000000_0111111_0000000
    assert mirror_player_board(board1) == board2


def test_add_mirror_to_dictionary():
    test_dict = {-1: {}}
    board1 = 0b_0100000_0000000_0000000_0000000_0000000_0000000_0000000
    board2 = 0b_0000000_0100000_0000000_0000000_0000000_0000000_0000000

    add_mirror_to_dictionary(board1, board2, test_dict, (1, [-1]), 0)

    mirror1, mirror2 = mirror_board(board1, board2)
    res = test_dict[mirror1][mirror2]
    assert res[0] == 1


def test_is_mirror_possible1():
    board1 = 0b_0100000_0000000_0000000_0000000_0000000_0000000_0000000
    board2 = 0b_0000000_0100000_0000000_0000000_0000000_0000000_0000000
    assert is_mirror_possible(board1, board2)


def test_is_mirror_possible2():
    board1 = 0b_0100000_0000000_0000000_0000000_0000000_0000000_0000000
    board2 = 0b_0000000_0100000_0000000_0000000_0000000_0000000_0100000
    assert not is_mirror_possible(board1, board2)
