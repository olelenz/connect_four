import pytest

from agents.game_utils import *


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

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == BoardPiece
    assert ret.shape == (6, 7)
    assert np.all(ret == NO_PLAYER)


def test_pretty_print_board():
    board = initialize_game_state()

    board[0, 0] = PLAYER1
    board[0, 1] = PLAYER2

    ret = pretty_print_board(board)

    assert isinstance(ret, str)
    assert ret == "|==============|\n|              |\n|              |\n|              |\n|              |\n|        " \
                  "      |\n|X O           |\n|==============|\n|0 1 2 3 4 5 6 |"


def test_string_to_board():
    board = initialize_game_state()

    board[0, 3] = PLAYER1
    board[0, 2] = PLAYER2

    ret = string_to_board(
        "|==============|\n|              |\n|              |\n|              |\n|              |\n|        "
        "      |\n|    O X       |\n|==============|\n|0 1 2 3 4 5 6 |")

    assert isinstance(ret, np.ndarray)
    assert (ret == board).all()

    with pytest.raises(AttributeError):
        string_to_board("wrong format")


def test_apply_player_action():
    board = initialize_game_state()
    board[0, 3] = PLAYER1
    board[1, 3] = PLAYER2
    board[2, 3] = PLAYER1
    board[3, 3] = PLAYER2
    board[4, 3] = PLAYER1

    ret = apply_player_action(board, PlayerAction(3), PLAYER2)

    assert isinstance(ret, np.ndarray)
    assert (ret != board).any()
    board[5, 3] = PLAYER2
    assert (ret == board).all()

    ret = apply_player_action(board, PlayerAction(2), PLAYER1)

    board[0, 2] = PLAYER1
    assert (ret == board).all()

    with pytest.raises(ValueError):
        apply_player_action(board, PlayerAction(3), PLAYER2)


def test_connected_four_general():
    ret = connected_four(initialize_game_state(), PLAYER1)
    assert not ret
    ret = connected_four(initialize_game_state(), PLAYER2)
    assert not ret


def test_connected_four_vertical():
    # vertical
    board_one = initialize_game_state()
    board_one[0, 3] = PLAYER1
    board_one[1, 3] = PLAYER1
    board_one[2, 3] = PLAYER1
    board_one[3, 3] = PLAYER1

    ret = connected_four(board_one, PLAYER1)
    assert ret
    ret = connected_four(board_one, PLAYER2)
    assert not ret


def test_connected_four_horizontal():
    # horizontal
    board_two = initialize_game_state()
    board_two[0, 0] = PLAYER2
    board_two[0, 1] = PLAYER2
    board_two[0, 2] = PLAYER2
    board_two[0, 3] = PLAYER2

    ret = connected_four(board_two, PLAYER2)
    assert ret
    ret = connected_four(board_two, PLAYER1)
    assert not ret


def test_connected_four_diagonal_one():
    # diagonal left top
    board_three = initialize_game_state()
    board_three[0, 0] = PLAYER1
    board_three[0, 1] = PLAYER1
    board_three[0, 2] = PLAYER2
    board_three[0, 3] = PLAYER1

    board_three[1, 0] = PLAYER2
    board_three[1, 1] = PLAYER2
    board_three[1, 2] = PLAYER1

    board_three[2, 0] = PLAYER2
    board_three[2, 1] = PLAYER1
    board_three[2, 2] = PLAYER2

    board_three[3, 0] = PLAYER1

    ret = connected_four(board_three, PLAYER1)
    assert ret
    ret = connected_four(board_three, PLAYER2)
    assert not ret


def test_connected_four_diagonal_two():
    # diagonal right top
    board_three = initialize_game_state()
    board_three[3, 0] = PLAYER1
    board_three[3, 1] = PLAYER1
    board_three[3, 2] = PLAYER2
    board_three[3, 3] = PLAYER1

    board_three[2, 0] = PLAYER2
    board_three[2, 1] = PLAYER2
    board_three[2, 2] = PLAYER1

    board_three[1, 0] = PLAYER2
    board_three[1, 1] = PLAYER1
    board_three[1, 2] = PLAYER2

    board_three[0, 0] = PLAYER1

    ret = connected_four(board_three, PLAYER1)
    assert ret
    ret = connected_four(board_three, PLAYER2)
    assert not ret


def test_check_end_state():
    board = initialize_game_state()

    board[0, 3] = PLAYER1
    board[0, 2] = PLAYER2

    ret = check_end_state(board, PLAYER1)

    assert isinstance(ret, GameState)
    assert ret == GameState.STILL_PLAYING

    board[0, 0:4] = PLAYER1

    ret = check_end_state(board, PLAYER1)

    assert isinstance(ret, GameState)
    assert ret == GameState.IS_WIN

    board = string_to_board(
        "|==============|\n|O X O X O X O |\n|X X O O O X X |\n|O O X X X O O |\n|O O O X O O O |\n|X X O O "
        "O X X |\n|X X X O O O X |\n|==============|\n|0 1 2 3 4 5 6 |")

    ret = check_end_state(board, PLAYER1)

    assert isinstance(ret, GameState)
    assert ret == GameState.IS_DRAW


def test_get_possible_moves():
    board = string_to_board("|==============|\n|O X O X O X O |\n|X X O O O X X |\n|O O X X X O O |\n|O O O X O O O |\n|X X O O O X X |\n|X X X O O O X |\n|==============|\n|0 1 2 3 4 5 6 |")
    ret = get_possible_moves(board)

    assert ret == []

    board = initialize_game_state()

    ret = get_possible_moves(board)

    assert isinstance(ret[0], PlayerAction)
    assert ret == [3, 2, 4, 1, 5, 0, 6]

    board[0:3, 3] = PLAYER1
    board[3:6, 3] = PLAYER2

    ret = get_possible_moves(board)

    assert isinstance(ret[0], PlayerAction)
    assert ret == [2, 4, 1, 5, 0, 6]
