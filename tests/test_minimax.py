def test_generate_move():
    from agents.agent_minimax import generate_move_minimax
    from agents.game_utils import initialize_game_state, PLAYER1

    board = initialize_game_state()
    ret = generate_move_minimax(board, PLAYER1, None, 2)


def test_evaluate_position():
    from agents.agent_minimax import evaluate_position
    from agents.game_utils import initialize_game_state, apply_player_action, PLAYER1,PLAYER2, PlayerAction

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

