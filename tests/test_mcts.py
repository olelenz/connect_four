def test_generate_move():
    from agents.agent_mcts import generate_move_mcts
    from agents.game_utils import initialize_game_state, PLAYER1

    board = initialize_game_state()
    ret = generate_move_mcts(board, PLAYER1, None, 2)
