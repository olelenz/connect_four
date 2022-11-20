from tests import test_game_utils, test_random, test_minimax, test_mcts


def test_all():
    test_game_utils.test_run_all()
    test_random.test_run_all()
    test_minimax.test_run_all()
    test_mcts.test_run_all()
