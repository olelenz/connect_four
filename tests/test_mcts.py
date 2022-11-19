from agents.agent_mcts import generate_move_mcts
from agents.game_utils import string_to_board, initialize_game_state, PLAYER2, PlayerAction, PLAYER1


def test_run_all():
    test_generate_move()

    test_win_in_one_move()
    test_prevent_opponent_win()
    test_win_in_two_moves()


def test_generate_move():
    from agents.agent_mcts import generate_move_mcts
    from agents.game_utils import initialize_game_state, PLAYER1

    #board = initialize_game_state()
    board = string_to_board(
        "|==============|\n|              |\n|              |\n|              |\n|              |\n|        "
        "      |\n|    O X X X   |\n|==============|\n|0 1 2 3 4 5 6 |")
    #board = string_to_board(
    #    "|==============|\n|              |\n|        O O O |\n|O O O   O O O |\n|O O O   O O O |\n|O O O   "
    #    "X X X |\n|X X X   O O O |\n|==============|\n|0 1 2 3 4 5 6 |")
    ret = generate_move_mcts(board, PLAYER1, None, 2)



def test_win_in_one_move():
    board = initialize_game_state()
    board[0, 0:3] = PLAYER2

    ret = generate_move_mcts(board, PLAYER2, None, 1)  # not deterministic...
    assert isinstance(ret[0], PlayerAction)
    assert ret[0] == 3


def test_prevent_opponent_win():
    board = initialize_game_state()
    board[0, 1:4] = PLAYER1
    board[0, 0] = PLAYER2

    ret = generate_move_mcts(board, PLAYER2, None, 1)  # not deterministic...
    assert isinstance(ret[0], PlayerAction)
    assert ret[0] == 4


def test_win_in_two_moves():
    board = initialize_game_state()
    board[0, 1] = PLAYER2
    board[0, 3] = PLAYER2

    ret = generate_move_mcts(board, PLAYER2, None, 5)  # not deterministic...
    assert isinstance(ret[0], PlayerAction)
    assert ret[0] == 2
