from agents.agent_minimax import generate_move_minimax, evaluate_position
from agents.game_utils import initialize_game_state, PLAYER1, apply_player_action, PlayerAction, PLAYER2


def test_run_all():
    test_generate_move()
    test_evaluate_position()
    test_player2_start()
    test_evaluate_winning_position_1()
    test_evaluate_winning_position_2()


def test_generate_move():
    board = initialize_game_state()
    ret = generate_move_minimax(board, PLAYER1, None, 2)

    assert isinstance(ret[0], PlayerAction)
    assert ret[0] in [0, 1, 2, 3, 4, 5, 6]


def test_evaluate_position():
    board = initialize_game_state()
    ret = evaluate_position(board)

    assert isinstance(ret, int)
    assert ret == 0

    board = apply_player_action(board, PlayerAction(0), PLAYER1)
    ret = evaluate_position(board)

    assert ret == 3

    board2 = initialize_game_state()
    board2 = apply_player_action(board2, PlayerAction(3), PLAYER1)
    board2 = apply_player_action(board2, PlayerAction(4), PLAYER2)
    ret = evaluate_position(board2)

    assert ret == 0


def test_player2_start():
    board = initialize_game_state()
    ret = generate_move_minimax(board, PLAYER2, None, 4)

    assert isinstance(ret[0], PlayerAction)
    assert ret[0] in [0, 1, 2, 3, 4, 5, 6]


def test_evaluate_winning_position_1():
    board = initialize_game_state()
    board[0:4] = PLAYER1
    ret = evaluate_position(board)

    assert isinstance(ret, int)
    assert ret == 1_000_000_000_000


def test_evaluate_winning_position_2():
    board = initialize_game_state()
    board[0:4] = PLAYER2
    ret = evaluate_position(board)

    assert isinstance(ret, int)
    assert ret == -1_000_000_000_000


