def test_run_all():
    test_generate_move_random()


def test_generate_move_random():
    from agents.agent_random import generate_move_random
    from agents.game_utils import string_to_board, PLAYER1, PlayerAction, initialize_game_state

    board_string = "|==============|\n|  O X O X O X |\n|X O X O X O X |\n|X O X O X O X |\n|O X O X O X O |\nO X O X " \
                   "O X O X |\n|X O X O X O X |\n|==============|\n|0 1 2 3 4 5 6 | "
    board = string_to_board(board_string)

    ret, _ = generate_move_random(board, PLAYER1, None)

    assert isinstance(ret, PlayerAction)
    assert ret == 0

    board = initialize_game_state()

    ret, _ = generate_move_random(board, PLAYER1, None)

    assert isinstance(ret, PlayerAction)
    assert ret in [0, 1, 2, 3, 4, 5, 6]
