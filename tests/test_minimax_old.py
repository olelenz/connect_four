import pytest

from agents.agent_minimax import generate_move_minimax, evaluate_position
from agents.agent_minimax.minimax_old import number_of_connected_n, empty_board_positions, evaluate_board_using_windows, \
    list_windows, evaluate_window
from agents.game_utils import initialize_game_state, PLAYER1, apply_player_action, PlayerAction, PLAYER2, \
    string_to_board, pretty_print_board


def test_run_all():
    test_generate_move()
    test_evaluate_position()
    test_player2_start()
    test_win_in_one_move()
    test_prevent_opponent_win()
    test_win_in_two_moves()
    test_game_losing_choose_most_moves_to_loss()


def test_generate_move():  # TODO: change to binary
    board = initialize_game_state()
    ret = generate_move_minimax(board, PLAYER1, None, 2)

    assert isinstance(ret[0], PlayerAction)
    assert ret[0] in [0, 1, 2, 3, 4, 5, 6]


def test_evaluate_position():
    board_player_one, board_player_two = initialize_game_state()
    ret = evaluate_position(board_player_one, board_player_two)

    assert isinstance(ret, int)
    assert ret == 0

    board_player_one, board_player_two = apply_player_action(board_player_one, board_player_two, PlayerAction(0),
                                                             PLAYER1)
    ret = evaluate_position(board_player_one, board_player_two)

    assert ret == 0

    board_player_one, board_player_two = initialize_game_state()
    board_player_one, board_player_two = apply_player_action(board_player_one, board_player_two, PlayerAction(3),
                                                             PLAYER1)
    board_player_one, board_player_two = apply_player_action(board_player_one, board_player_two, PlayerAction(4),
                                                             PLAYER2)
    ret = evaluate_position(board_player_one, board_player_two)

    assert ret == 0


def test_minimax_player_two_start():
    board_player_one = 0b0000000_0000000_0000000_0000000_0000000_0000000_0000000
    board_player_two = 0b0000000_0000000_0000000_0000000_0000000_0000000_0000000

    ret = generate_move_minimax(board_player_one, board_player_two, PLAYER2, None, 10)
    assert isinstance(ret[0], PlayerAction)
    assert ret[0] in [0, 1, 2, 3, 4, 5, 6]


def test_avoid_loss_transposition_table():
    board_player_one = 0b0000000_0000000_0011100_0010001_0001011_0001011_0000000
    board_player_two = 0b0000000_0000001_0000011_0001110_0010100_0000100_0000001
    print(pretty_print_board(board_player_one, board_player_two))

    ret = generate_move_minimax(board_player_one, board_player_two, PLAYER2, None, 21)
    assert isinstance(ret[0], PlayerAction)
    assert ret[0] == 4


def test_play_valid_move():
    board_player_one = 0b101010101101101100010101000000100010000000110
    board_player_two = 0b11010001010000010010101010000000001101110001001

    ret = generate_move_minimax(board_player_one, board_player_two, PLAYER2, None, 21)
    assert isinstance(ret[0], PlayerAction)
    assert ret[0] in [0, 1, 2, 3, 4, 5, 6]


def test_win_in_one_move():
    board_player_one = 0b0000000_0000000_0000000_0000000_0000000_0000000_0000000
    board_player_two = 0b0000000_0000000_0000000_0000000_0000111_0000000_0000000

    ret = generate_move_minimax(board_player_one, board_player_two, PLAYER2, None, 2)
    assert isinstance(ret[0], PlayerAction)
    assert ret[0] == 2


def test_prevent_opponent_win():
    board_player_one = 0b0000000_0000000_0000000_0000001_0000001_0000001_0000000
    board_player_two = 0b0000000_0000000_0000000_0000000_0000000_0000000_0000001

    ret = generate_move_minimax(board_player_one, board_player_two, PLAYER2, None, 2)
    assert isinstance(ret[0], PlayerAction)
    assert ret[0] == 4


def test_prevent_opponent_win_two():
    board_player_one = 0b0000000_0000000_0000000_0110101_0000001_0000000_0000000
    board_player_one = 0b110101_0000001_0000000_0000000

    board_player_two = 0b0000000_0000111_0000000_0001010_0000000_0000000_0000000
    board_player_two = 0b11100000000001010000000000000000000000

    ret = generate_move_minimax(board_player_one, board_player_two, PLAYER1, None, 7)
    assert isinstance(ret[0], PlayerAction)
    assert ret[0] == 5


def test_win_in_two_moves():
    board_player_one = 0b0000000_0000000_0000000_0000000_0000000_0000000_0000000
    board_player_two = 0b0000000_0000000_0000000_0000001_0000000_0000001_0000000

    ret = generate_move_minimax(board_player_one, board_player_two, PLAYER2, None, 3)
    assert isinstance(ret[0], PlayerAction)
    assert ret[0] == 2


def test_game_losing_choose_most_moves_to_loss():  # TODO: change to binary
    board = string_to_board(
        "|==============|\n|              |\n|            O |\n|          X X |\n|      O   X X |\n|      O O X O |\n|O     X X O X |\n|==============|\n|0 1 2 3 4 5 6 |")

    ret = generate_move_minimax(board, PLAYER2, None, 4)
    assert isinstance(ret[0], PlayerAction)
    assert ret[0] == 5


def test_number_of_connected_n():
    board = 0b0000000_0000000_0000000_0000111_0000000_0000000_0000000

    ret = number_of_connected_n(board, 2)
    assert isinstance(ret, int)
    assert ret == 2

    ret = number_of_connected_n(board, 3)
    assert ret == 1

    board = 0b0000000_0000000_0000010_0000001_0000001_0000001_0000000

    with pytest.raises(AssertionError):
        number_of_connected_n(board, 0)

    ret = number_of_connected_n(board, 2)
    assert ret == 3

    ret = number_of_connected_n(board, 3)
    assert ret == 1

    board = 0b0000000_0000000_0000000_0000000_0000000_0000000_0000001
    ret = number_of_connected_n(board, 1)
    assert ret == 4


def test_evaluate_position_two():
    board_player_one = 0b0000000_0000000_0100000_0000000_0000000_0000000_0000000
    board_player_two = 0b0011111_0011111_0011111_0011111_0011111_0011111_0011111
    ret = evaluate_position(board_player_one, board_player_two)
    assert isinstance(ret, int)
    assert ret == 4


def test_empty_board_positions():
    board_1 = 0b0100000_0000000_0000000_0110000_0000000_0100000_0000000
    board_2 = 0b0000000_0100000_0100000_0000000_0110000_0000000_0100000
    expected_result = 0b0011111_0011111_0011111_0001111_0001111_0011111_0011111

    assert empty_board_positions(board_1, board_2) == expected_result


def test_evaluate_board_using_windows1():
    board_1 = 0b0100000_0000000_0000000_0000000_0000000_0000000_0000000
    board_2 = 0b0000000_0000000_0000000_0000000_0000000_0000000_0100000
    res = evaluate_board_using_windows(board_1, board_2)
    assert res == 0


def test_evaluate_board_using_windows2():
    board_1 = 0b0100000_0000000_0000000_0000000_0000000_0000000_0000000
    board_2 = 0b0000000_0000000_0000000_0000000_0000000_0000000_0000000
    res = evaluate_board_using_windows(board_1, board_2)
    assert res == 3


def test_evaluate_board_using_windows3():
    board_1 = 0b0111000_0000000_0000000_0000000_0000000_0000000_0000000  # 10 2 1 1 1 1 1 1 1 = 19 points
    board_2 = 0b0000000_0000000_0000000_0000000_0000000_0000000_0100000  # 3 points (negative)
    res = evaluate_board_using_windows(board_1, board_2)
    assert res == 16


def test_list_windows():
    res = list_windows()
    assert len(res) == 69


def test_evaluate_window():
    window_position1 = 0b0000000_0000000_0000000_0000000_0000000_0000000_0000001
    window_position2 = 0b0000000_0000000_0000000_0000000_0000000_0000001_0000000
    window_position3 = 0b0000000_0000000_0000000_0000000_0000001_0000000_0000000
    window_position4 = 0b0000000_0000000_0000000_0000001_0000000_0000000_0000000
    board_player1 = 0b0100000_0000000_0000000_0000000_0000000_0000000_0000000
    board_player2 = 0b0000000_0100000_0000000_0000000_0000000_0000000_0000000
    res = evaluate_window((window_position1, window_position2, window_position3, window_position4), board_player1,
                          board_player2)
    assert res == 0

    board_player2 = 0b0000000_0100000_0000000_0000000_0000000_0000001_0000001
    res = evaluate_window((window_position1, window_position2, window_position3, window_position4), board_player1,
                          board_player2)
    assert res == -2

    board_player2 = 0b0000000_0100000_0000000_0000000_0000001_0000001_0000001
    res = evaluate_window((window_position1, window_position2, window_position3, window_position4), board_player1,
                          board_player2)
    assert res == -10