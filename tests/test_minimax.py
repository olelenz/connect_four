import pytest

from agents.agent_minimax.minimax import handle_empty_moves_eval, START_VALUE, get_possible_moves_iterative, \
    mirror_boards, mirror_player_board, add_mirrored_boards_to_dictionary, use_mirror_functions, \
    evaluate_board_using_windows, evaluate_window, calculate_evaluation_score, get_eval_from_dictionary
from agents.game_utils import *
from agents.agent_minimax.minimax_window_list import MINIMAX_EVALUATION_WINDOWS_LIST, list_windows


EMPTY_BOARD: int = 0b0000000_0000000_0000000_0000000_0000000_0000000_0000000
DRAW_PLAYER_ONE: int = 0b0001010_0010101_0111011_0101110_0000100_0010100_0100011
DRAW_PLAYER_TWO: int = 0b0110101_0101010_0000100_0010001_0111011_0101011_0011100
DIAGONAL_BOARD_LEFT_TOP: int = 0b0000000_0000000_0000001_0000010_0000100_0001000_0000000

EXAMPLE_BOARD: int = 0b0000000_0001011_0000110_0000010_0000001_0000000_0000100
MIRRORED_EXAMPLE_BOARD: int = 0b0000100_0000000_0000001_0000010_0000110_0001011_0000000

LEFT_TOWER_ONE_BOARD: int = 0b0000000_0000000_0000000_0000000_0000000_0000000_0000001
LEFT_TOWER_TWO_BOARD: int = 0b0000000_0000000_0000000_0000000_0000000_0000000_0000010

MIDDLE_TOWER_ONE_BOARD: int = 0b0000000_0000000_0000000_0000001_0000000_0000000_0000000
MIDDLE_TOWER_TWO_BOARD: int = 0b0000000_0000000_0000000_0000010_0000000_0000000_0000000

RIGHT_TOWER_ONE_BOARD: int = 0b0000001_0000000_0000000_0000000_0000000_0000000_0000000
RIGHT_TOWER_TWO_BOARD: int = 0b0000010_0000000_0000000_0000000_0000000_0000000_0000000

RIGHT_TOWER_TWO_IN_A_ROW: int = 0b0000011_0000000_0000000_0000000_0000000_0000000_0000000
RIGHT_TOWER_THREE_IN_A_ROW: int = 0b0000111_0000000_0000000_0000000_0000000_0000000_0000000

TEST_WINDOW_RIGHT_TOWER: (int, int, int, int) = (0b0000001_0000000_0000000_0000000_0000000_0000000_0000000,
                                                 0b0000010_0000000_0000000_0000000_0000000_0000000_0000000,
                                                 0b0000100_0000000_0000000_0000000_0000000_0000000_0000000,
                                                 0b0001000_0000000_0000000_0000000_0000000_0000000_0000000)

EXAMPLE_DICTIONARY_ENTRY: [int, [int]] = [1, [1, 2, 3]]

def test_handle_empty_moves_eval_draw():
    ret = handle_empty_moves_eval(PLAYER1, GameState.IS_DRAW, 4)
    assert ret == 0


def test_handle_empty_moves_eval_still_playing():
    with pytest.raises(AttributeError):
        handle_empty_moves_eval(PLAYER1, GameState.STILL_PLAYING, 4)


def test_handle_empty_moves_eval_player_one_won():
    ret = handle_empty_moves_eval(PLAYER1, GameState.IS_WIN, 4)
    assert ret == (-START_VALUE * 2 ** 4)


def test_handle_empty_moves_eval_player_two_won():
    ret = handle_empty_moves_eval(PLAYER2, GameState.IS_WIN, 5)
    assert ret == (START_VALUE * 2 ** 5)


def test_get_possible_moves_iterative_draw():
    ret_actions, game_state = get_possible_moves_iterative((DRAW_PLAYER_ONE, DRAW_PLAYER_TWO, PLAYER1), [2])
    assert ret_actions == []
    assert game_state == GameState.IS_DRAW


def test_get_possible_moves_iterative_win_player_one():
    ret_actions, game_state = get_possible_moves_iterative((DIAGONAL_BOARD_LEFT_TOP, EMPTY_BOARD, PLAYER2), [2])
    assert ret_actions == []
    assert game_state == GameState.IS_WIN


def test_get_possible_moves_iterative_win_player_two():
    ret_actions, game_state = get_possible_moves_iterative((EMPTY_BOARD, DIAGONAL_BOARD_LEFT_TOP, PLAYER1), [2])
    assert ret_actions == []
    assert game_state == GameState.IS_WIN


def test_get_possible_moves_iterative_empty_next_moves():
    ret_actions, game_state = get_possible_moves_iterative((EMPTY_BOARD, EMPTY_BOARD, PLAYER1), [])
    moves: list[PlayerAction] = MOVE_ORDER.copy()
    assert ret_actions == moves
    assert game_state == GameState.STILL_PLAYING


def test_get_possible_moves_iterative_full_next_moves():
    ret_actions, game_state = get_possible_moves_iterative((EMPTY_BOARD, EMPTY_BOARD, PLAYER1), [5])
    moves: list[PlayerAction] = MOVE_ORDER.copy()
    moves.remove(PlayerAction(5))
    moves = [PlayerAction(5)]+moves
    assert ret_actions == moves
    assert game_state == GameState.STILL_PLAYING


def test_generate_move_minimax():
    pass


def test_generate_move_loop_to_stop():
    pass


def test_generate_move_minimax_id():
    pass


def test_minimax_rec():
    pass


def test_get_alpha():
    pass


def test_get_beta():
    pass


def test_get_eval_from_dictionary_not_existing():
    ret = get_eval_from_dictionary(EMPTY_BOARD, EMPTY_BOARD, {})
    assert ret is None


def test_get_eval_from_dictionary_entry_existing():
    dictionary = {EMPTY_BOARD: {EMPTY_BOARD: EXAMPLE_DICTIONARY_ENTRY}}
    ret = get_eval_from_dictionary(EMPTY_BOARD, EMPTY_BOARD, dictionary)
    assert ret == EXAMPLE_DICTIONARY_ENTRY


def test_list_windows():
    ret = list_windows()
    assert ret == MINIMAX_EVALUATION_WINDOWS_LIST


def test_mirror_player_board():
    ret = mirror_player_board(EXAMPLE_BOARD)
    assert ret == MIRRORED_EXAMPLE_BOARD


def test_mirror_boards():
    ret = mirror_boards(EXAMPLE_BOARD, MIRRORED_EXAMPLE_BOARD)
    assert ret == (MIRRORED_EXAMPLE_BOARD, EXAMPLE_BOARD)


def test_add_mirrored_boards_to_dictionary():
    dictionary = {-1: {}}
    add_mirrored_boards_to_dictionary(LEFT_TOWER_ONE_BOARD, LEFT_TOWER_TWO_BOARD, dictionary, [10, [1, 1, 2]], 1)
    # mirrored boards should be in the dictionary, evaluation and mirrored move list
    ret = dictionary[RIGHT_TOWER_ONE_BOARD][RIGHT_TOWER_TWO_BOARD]
    assert ret == [10, [5, 5, 4]]


def test_use_mirror_functions_one():
    assert use_mirror_functions(EMPTY_BOARD, EMPTY_BOARD)


def test_use_mirror_functions_two():
    assert use_mirror_functions(MIDDLE_TOWER_ONE_BOARD, MIDDLE_TOWER_TWO_BOARD)


def test_use_mirror_functions_three():
    assert not use_mirror_functions(LEFT_TOWER_ONE_BOARD, RIGHT_TOWER_ONE_BOARD)


def test_evaluate_board_using_windows_one():
    res = evaluate_board_using_windows(LEFT_TOWER_ONE_BOARD, RIGHT_TOWER_ONE_BOARD)
    assert res == 0


def test_evaluate_board_using_windows_two():
    res = evaluate_board_using_windows(LEFT_TOWER_ONE_BOARD, EMPTY_BOARD)
    assert res == 3


def test_evaluate_board_using_windows_three():
    res = evaluate_board_using_windows(RIGHT_TOWER_THREE_IN_A_ROW, LEFT_TOWER_ONE_BOARD)
    # explanation: Player 1: 3-window: 6 points, 2-window: 4 points, 7x 1-window: 7 points = 17 points
    # Player 2: 3x 1-window: -3 points -> 17 - 3 = 14
    assert res == 14


def test_evaluate_window_one():
    res = evaluate_window(TEST_WINDOW_RIGHT_TOWER, RIGHT_TOWER_ONE_BOARD, LEFT_TOWER_ONE_BOARD)
    assert res == 1


def test_evaluate_window_two():
    res = evaluate_window(TEST_WINDOW_RIGHT_TOWER, RIGHT_TOWER_ONE_BOARD, RIGHT_TOWER_TWO_BOARD)
    assert res == 0


def test_evaluate_window_three():
    res = evaluate_window(TEST_WINDOW_RIGHT_TOWER, RIGHT_TOWER_TWO_IN_A_ROW, LEFT_TOWER_ONE_BOARD)
    assert res == 4


def test_evaluate_window_four():
    res = evaluate_window(TEST_WINDOW_RIGHT_TOWER, RIGHT_TOWER_THREE_IN_A_ROW, LEFT_TOWER_ONE_BOARD)
    assert res == 6


def test_calculate_evaluation_score_one():
    res = calculate_evaluation_score(1, 0)
    assert res == 1


def test_calculate_evaluation_score_two():
    res = calculate_evaluation_score(0, 2)
    assert res == -4


def test_calculate_evaluation_score_three():
    res = calculate_evaluation_score(3, 0)
    assert res == 6


def test_calculate_evaluation_score_four():
    res = calculate_evaluation_score(1, 2)
    assert res == 0


def test_ideas():
    print("hello")
    '''
    DONE - current depth und desired depth zu einem parameter -> auf null testesn und immer um eins verringern
    - two boards into one tuple with boards and current player
    - alpha and beta into one tuple
    - remove flags maximize and use_mirror
    - keep dictionary
    - moves_line and next_moves into one tuple
    --> five parameters instead of 12
    
    - loosen and part maximize and minimize part form rec function
    
    - loosen no more moves part from function
    
    DONE - keep anchor
    
    DONE - return IS_WIN in get_possible_moves function
    
    
    '''
