def test_generate_move():
    from agents.agent_minimax import generate_move_minimax
    from agents.game_utils import initialize_game_state, PLAYER1

    board = initialize_game_state()
    ret = generate_move_minimax(board, PLAYER1, None, 2)

def test_evaluate_position():
    from agents.agent_minimax import evaluate_position
    from agents.game_utils import initialize_game_state, PLAYER1

    board = initialize_game_state()
    ret = evaluate_position(board, PLAYER1)

    assert isinstance(ret[0], int)
    assert ret[0] == 0
