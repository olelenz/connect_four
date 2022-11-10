from agents.game_utils import string_to_board


def test_generate_move():
    from agents.agent_mcts import generate_move_mcts
    from agents.game_utils import initialize_game_state, PLAYER1

    #board = initialize_game_state()
    #board = string_to_board(
    #    "|==============|\n|              |\n|              |\n|              |\n|              |\n|        "
    #    "      |\n|    O X X X   |\n|==============|\n|0 1 2 3 4 5 6 |")
    board = string_to_board(
        "|==============|\n|              |\n|        O O O |\n|O O O   O O O |\n|O O O   O O O |\n|O O O   "
        "X X X |\n|X X X   O O O |\n|==============|\n|0 1 2 3 4 5 6 |")
    ret = generate_move_mcts(board, PLAYER1, None, 2)


