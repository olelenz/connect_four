import pytest

from agents.agent_mcts.mcts import selection, expansion, simulation, backpropagation, generate_move_mcts
from agents.agent_mcts.mcts_tree import MctsTree
from agents.game_utils import initialize_game_state, PLAYER2, PlayerAction, PLAYER1, apply_player_action, \
    pretty_print_board
from agents.saved_state import SavedState


def test_run_all():
    # are allowed to fail:
    test_win_in_one_move()
    test_prevent_opponent_win()
    test_win_in_two_moves()

    # should be always correct
    test_selection()
    test_expansion()
    test_simulation()
    test_backpropagation()
    test_use_saved_state()
    test_use_saved_state_no_children()
    test_no_calculation_time()


def test_win_in_one_move():
    board = initialize_game_state()
    board[0, 0:3] = PLAYER2

    ret = generate_move_mcts(board, PLAYER2, None, 5)  # not deterministic...
    assert isinstance(ret[0], PlayerAction)
    assert ret[0] == 3


def test_prevent_opponent_win():
    board = initialize_game_state()
    board[0, 1:4] = PLAYER1
    board[0, 0] = PLAYER2

    ret = generate_move_mcts(board, PLAYER2, None, 5)  # not deterministic...
    assert isinstance(ret[0], PlayerAction)
    assert ret[0] == 4


def test_win_in_two_moves():
    board = initialize_game_state()
    board[0, 1] = PLAYER2
    board[0, 3] = PLAYER2

    ret = generate_move_mcts(board, PLAYER2, None, 5)  # not deterministic...
    assert isinstance(ret[0], PlayerAction)
    assert ret[0] == 2


def test_selection():
    board = initialize_game_state()
    starting_node = MctsTree(None, board, None, PLAYER1)
    starting_node.increment_n()
    board_list = []
    for i in range(7):
        board_list.append(MctsTree(starting_node, apply_player_action(board, PlayerAction(i), PLAYER1), PlayerAction(i), PLAYER1))
        board_list[i].increment_n()
        board_list[i].update_w(i)
        if i == 4:
            board_list[i].update_w(100)
        starting_node.add_child_tree(board_list[i])
    ret = selection(starting_node)

    assert isinstance(ret, MctsTree)
    assert pretty_print_board(ret.get_board()) == pretty_print_board(board_list[4].get_board())


def test_expansion():
    board = initialize_game_state()
    starting_node = MctsTree(None, board, None, PLAYER1)
    comp_node = MctsTree(None, board, None, PLAYER1)
    starting_node.increment_n()
    for i in range(7):
        new_tree = MctsTree(starting_node, apply_player_action(board, PlayerAction(i), PLAYER1), PlayerAction(i), PLAYER1)
        starting_node.add_child_tree(new_tree)
    expansion(comp_node)

    for i, tree in enumerate(comp_node.get_child_trees()):
        assert pretty_print_board(tree.get_board()) == pretty_print_board(starting_node.get_child_trees()[i].get_board())


def test_simulation():
    ret = simulation(initialize_game_state(), PLAYER1, PLAYER1)
    assert isinstance(ret, int)
    assert ret in [-1, 0, 1]


def test_backpropagation():
    board = initialize_game_state()
    root = MctsTree(None, board, None, PLAYER1)
    root.update_w(3)
    root.increment_n()
    root.add_child_tree(MctsTree(root, apply_player_action(board, PlayerAction(1), PLAYER1), PlayerAction(1), PLAYER1))
    backpropagation(root.get_child_trees()[0], 53)

    assert root.get_child_trees()[0].get_w() == 53
    assert root.get_child_trees()[0].get_n() == 1
    assert root.get_w() == 56
    assert root.get_n() == 2


def test_use_saved_state():
    board = initialize_game_state()
    starting_node = MctsTree(None, board, None, PLAYER1)
    starting_node.increment_n()
    for i in range(7):
        new_tree = MctsTree(starting_node, apply_player_action(board, PlayerAction(i), PLAYER1), PlayerAction(i),
                            PLAYER1)
        starting_node.add_child_tree(new_tree)

    saved_state = SavedState(starting_node)

    ret = generate_move_mcts(apply_player_action(board, PlayerAction(2), PLAYER1), PLAYER2, saved_state, 1)
    assert isinstance(ret[0], PlayerAction)
    assert ret[0] in [0, 1, 2, 3, 4, 5, 6]


def test_use_saved_state_no_children():
    board = initialize_game_state()
    starting_node = MctsTree(None, board, None, PLAYER1)
    starting_node.increment_n()
    saved_state = SavedState(starting_node)

    ret = generate_move_mcts(apply_player_action(board, PlayerAction(2), PLAYER1), PLAYER2, saved_state, 1)
    assert isinstance(ret[0], PlayerAction)
    assert ret[0] in [0, 1, 2, 3, 4, 5, 6]


def test_no_calculation_time():
    with pytest.raises(RuntimeError):
        generate_move_mcts(apply_player_action(initialize_game_state(), PlayerAction(2), PLAYER1), PLAYER2, None, 0)
