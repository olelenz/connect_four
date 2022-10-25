import numpy as np
from agents.game_utils import BoardPiece, NO_PLAYER, PLAYER1, PLAYER2


def test_initialize_game_state():
    from agents.game_utils import initialize_game_state

    ret = initialize_game_state()

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == BoardPiece
    assert ret.shape == (6, 7)
    assert np.all(ret == NO_PLAYER)


def test_pretty_print_board():
    from agents.game_utils import initialize_game_state, pretty_print_board

    board = initialize_game_state()

    board[0, 0] = PLAYER1
    board[0, 1] = PLAYER2

    ret = pretty_print_board(board)

    assert isinstance(ret, str)
    assert ret == "|==============|\n|              |\n|              |\n|              |\n|              |\n|        " \
                  "      |\n|X O           |\n|==============|\n|0 1 2 3 4 5 6 |"


def test_string_to_board():
    from agents.game_utils import initialize_game_state, string_to_board

    board = initialize_game_state()

    board[0, 3] = PLAYER1
    board[0, 2] = PLAYER2

    ret = string_to_board("|==============|\n|              |\n|              |\n|              |\n|              |\n|        " \
                  "      |\n|    O X       |\n|==============|\n|0 1 2 3 4 5 6 |")

    assert isinstance(ret, np.ndarray)
    assert (ret == board).all()
