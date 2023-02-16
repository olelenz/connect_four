import pytest
from agents.game_utils import *

EMPTY_BOARD: int = 0b0000000_0000000_0000000_0000000_0000000_0000000_0000000
EMPTY_BOARD_STRING: str = "|==============|\n|              |\n|              |\n|              |\n|              |\n|              |\n|              |\n|==============|\n|0 1 2 3 4 5 6 |"
COLUMN_TWO_FILLED_BOARD_PLAYER_ONE: int = 0b0000000_0000000_0000000_0000000_0010101_0000000_0000000
COLUMN_TWO_FILLED_BOARD_PLAYER_TWO: int = 0b0000000_0000000_0000000_0000000_0101010_0000000_0000000
COLUMN_TWO_FILLED_STRING: str = "|==============|\n|    O         |\n|    X         |\n|    O         |\n|    X         |\n|    O         |\n|    X         |\n|==============|\n|0 1 2 3 4 5 6 |"
FIRST_PIECE_BOARD: int = 0b0000000_0000000_0000000_0000001_0000000_0000000_0000000
FIRST_PIECE_STRING: str = "|==============|\n|              |\n|              |\n|              |\n|              |\n|              |\n|      X       |\n|==============|\n|0 1 2 3 4 5 6 |"
LEFT_BOTTOM_CORNER_PIECE_BOARD: int = 0b0000000_0000000_0000000_0000000_0000000_0000000_0000001
LEFT_BOTTOM_CORNER_PIECE_STRING: str = "|==============|\n|              |\n|              |\n|              |\n|              |\n|              |\n|X             |\n|==============|\n|0 1 2 3 4 5 6 |"
DIAGONAL_BOARD_RIGHT_TOP: int = 0b0000000_0000000_0000000_0001000_0000100_0000010_0000001
DIAGONAL_BOARD_RIGHT_TOP_PLAYER_ONE_STRING: str = "|==============|\n|              |\n|              |\n|      X       |\n|    X         |\n|  X           |\n|X             |\n|==============|\n|0 1 2 3 4 5 6 |"
DIAGONAL_BOARD_RIGHT_TOP_PLAYER_TWO_STRING: str = "|==============|\n|              |\n|              |\n|      O       |\n|    O         |\n|  O           |\n|O             |\n|==============|\n|0 1 2 3 4 5 6 |"

LEFT_TOWER_ONE_BOARD: int = 0b0000000_0000000_0000000_0000000_0000000_0000000_0000001
LEFT_TOWER_TWO_BOARD: int = 0b0000000_0000000_0000000_0000000_0000000_0000000_0000010
LEFT_TOWER_THREE_BOARD: int = 0b0000000_0000000_0000000_0000000_0000000_0000000_0000101
LEFT_TOWER_FOUR_BOARD: int = 0b0000000_0000000_0000000_0000000_0000000_0000000_0001010
LEFT_TOWER_FIVE_BOARD: int = 0b0000000_0000000_0000000_0000000_0000000_0000000_0010101
LEFT_TOWER_SIX_BOARD: int = 0b0000000_0000000_0000000_0000000_0000000_0000000_0101010

RIGHT_TOWER_ONE_BOARD: int = 0b0000001_0000000_0000000_0000000_0000000_0000000_0000000
RIGHT_TOWER_TWO_BOARD: int = 0b0000010_0000000_0000000_0000000_0000000_0000000_0000000

DIAGONAL_BOARD_LEFT_TOP: int = 0b0000000_0000000_0000001_0000010_0000100_0001000_0000000
HORIZONTAL_BOARD: int = 0b0000000_0000000_0000000_0001000_0001000_0001000_0001000
VERTICAL_BOARD: int = 0b0000000_0001111_0000000_0000000_0000000_0000000_0000000
FULL_WIN_BOARD: int = 0b0000000_0001011_0000110_0000010_0000001_0000000_0000100
FULL_NO_WIN_BOARD: int = 0b0000000_0001011_0000110_0000000_0000001_0000000_0000100
DRAW_PLAYER_ONE: int = 0b0001010_0010101_0111011_0101110_0000100_0010100_0100011
DRAW_PLAYER_TWO: int = 0b0110101_0101010_0000100_0010001_0111011_0101011_0011100


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


def test_string_to_board_assertion_error():
    with pytest.raises(AttributeError):
        string_to_board(EMPTY_BOARD_STRING+"\n")


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


def test_apply_player_action_empty_board_player_one():
    ret = apply_player_action((EMPTY_BOARD, EMPTY_BOARD, PLAYER1), PlayerAction(3))
    assert ret[0] == FIRST_PIECE_BOARD
    assert ret[1] == EMPTY_BOARD


def test_apply_player_action_empty_board_player_two():
    ret = apply_player_action((EMPTY_BOARD, EMPTY_BOARD, PLAYER2), PlayerAction(3))
    assert ret[0] == EMPTY_BOARD
    assert ret[1] == FIRST_PIECE_BOARD


def test_apply_player_action_left_bottom_corner_player_one():
    ret = apply_player_action((EMPTY_BOARD, EMPTY_BOARD, PLAYER1), PlayerAction(0))
    assert ret[0] == LEFT_BOTTOM_CORNER_PIECE_BOARD
    assert ret[1] == EMPTY_BOARD


def test_apply_player_action_left_bottom_corner_player_two():
    ret = apply_player_action((EMPTY_BOARD, EMPTY_BOARD, PLAYER2), PlayerAction(0))
    assert ret[0] == EMPTY_BOARD
    assert ret[1] == LEFT_BOTTOM_CORNER_PIECE_BOARD


def test_apply_player_action_column_two_full():
    with pytest.raises(ValueError):
        apply_player_action((COLUMN_TWO_FILLED_BOARD_PLAYER_ONE, COLUMN_TWO_FILLED_BOARD_PLAYER_TWO, PLAYER1), PlayerAction(2))


def test_apply_player_action_row_two():
    ret = apply_player_action((LEFT_TOWER_ONE_BOARD, EMPTY_BOARD, PLAYER2), PlayerAction(0))
    assert ret[0] == LEFT_TOWER_ONE_BOARD
    assert ret[1] == LEFT_TOWER_TWO_BOARD


def test_apply_player_action_row_three():
    ret = apply_player_action((LEFT_TOWER_ONE_BOARD, LEFT_TOWER_TWO_BOARD, PLAYER1), PlayerAction(0))
    assert ret[0] == LEFT_TOWER_THREE_BOARD
    assert ret[1] == LEFT_TOWER_TWO_BOARD


def test_apply_player_action_row_four():
    ret = apply_player_action((LEFT_TOWER_THREE_BOARD, LEFT_TOWER_TWO_BOARD, PLAYER2), PlayerAction(0))
    assert ret[0] == LEFT_TOWER_THREE_BOARD
    assert ret[1] == LEFT_TOWER_FOUR_BOARD


def test_apply_player_action_row_five():
    ret = apply_player_action((LEFT_TOWER_THREE_BOARD, LEFT_TOWER_FOUR_BOARD, PLAYER1), PlayerAction(0))
    assert ret[0] == LEFT_TOWER_FIVE_BOARD
    assert ret[1] == LEFT_TOWER_FOUR_BOARD


def test_apply_player_action_row_six():
    ret = apply_player_action((LEFT_TOWER_FIVE_BOARD, LEFT_TOWER_FOUR_BOARD, PLAYER2), PlayerAction(0))
    assert ret[0] == LEFT_TOWER_FIVE_BOARD
    assert ret[1] == LEFT_TOWER_SIX_BOARD


def test_connected_four_empty_board():
    ret = connected_four(EMPTY_BOARD)
    assert not ret


def test_connected_four_horizontal():
    ret = connected_four(HORIZONTAL_BOARD)
    assert ret


def test_connected_four_vertical():
    ret = connected_four(VERTICAL_BOARD)
    assert ret


def test_connected_four_diagonal_right_top():
    ret = connected_four(DIAGONAL_BOARD_RIGHT_TOP)
    assert ret


def test_connected_four_diagonal_left_top():
    ret = connected_four(DIAGONAL_BOARD_LEFT_TOP)
    assert ret


def test_connected_four_many_pieces_no_win():
    ret = connected_four(FULL_NO_WIN_BOARD)
    assert not ret


def test_connected_four_many_pieces_win():
    ret = connected_four(FULL_WIN_BOARD)
    assert ret


def test_check_end_state_still_playing():
    ret = check_end_state(LEFT_TOWER_TWO_BOARD, LEFT_TOWER_ONE_BOARD, PLAYER1)
    assert isinstance(ret, GameState)
    assert ret == GameState.STILL_PLAYING


def test_check_end_state_draw():
    ret = check_end_state(DRAW_PLAYER_ONE, DRAW_PLAYER_TWO, PLAYER1)
    assert isinstance(ret, GameState)
    assert ret == GameState.IS_DRAW


def test_check_end_state_is_win_player_one():
    ret = check_end_state(FULL_WIN_BOARD, EMPTY_BOARD, PLAYER1)
    assert isinstance(ret, GameState)
    assert ret == GameState.IS_WIN


def test_check_end_state_is_win_player_two():
    ret = check_end_state(EMPTY_BOARD, FULL_WIN_BOARD, PLAYER2)
    assert isinstance(ret, GameState)
    assert ret == GameState.IS_WIN


def test_get_possible_moves_all_player_one():
    ret = get_possible_moves(EMPTY_BOARD, EMPTY_BOARD, PLAYER1)
    assert ret == (MOVE_ORDER, GameState.STILL_PLAYING)


def test_get_possible_moves_all_player_two():
    ret = get_possible_moves(EMPTY_BOARD, EMPTY_BOARD, PLAYER2)
    assert ret == (MOVE_ORDER, GameState.STILL_PLAYING)


def test_get_possible_moves_none_player_one():
    ret = get_possible_moves(DRAW_PLAYER_ONE, DRAW_PLAYER_TWO, PLAYER1)
    assert ret == ([], GameState.IS_DRAW)


def test_get_possible_moves_none_player_two():
    ret = get_possible_moves(DRAW_PLAYER_ONE, DRAW_PLAYER_TWO, PLAYER2)
    assert ret == ([], GameState.IS_DRAW)


def test_get_possible_moves_column_two_full_player_one():
    ret = get_possible_moves(COLUMN_TWO_FILLED_BOARD_PLAYER_ONE, COLUMN_TWO_FILLED_BOARD_PLAYER_TWO, PLAYER1)
    moves: list[PlayerAction] = MOVE_ORDER.copy()
    moves.remove(PlayerAction(2))
    assert ret == (moves, GameState.STILL_PLAYING)


def test_get_possible_moves_column_two_full_player_two():
    ret = get_possible_moves(COLUMN_TWO_FILLED_BOARD_PLAYER_ONE, COLUMN_TWO_FILLED_BOARD_PLAYER_TWO, PLAYER2)
    moves: list[PlayerAction] = MOVE_ORDER.copy()
    moves.remove(PlayerAction(2))
    assert ret == (moves, GameState.STILL_PLAYING)


def test_get_possible_moves_player_one_won():
    ret = get_possible_moves(HORIZONTAL_BOARD, EMPTY_BOARD, PLAYER2)
    assert ret == ([], GameState.IS_WIN)


def test_get_possible_moves_player_two_won():
    ret = get_possible_moves(EMPTY_BOARD, DIAGONAL_BOARD_RIGHT_TOP, PLAYER1)
    assert ret == ([], GameState.IS_WIN)
