def test_generate_move():
    from agents.agent_minimax import generate_move
    from agents.game_utils import initialize_game_state, PLAYER1

    board = initialize_game_state()
    ret = generate_move(board, PLAYER1, None, 2)
