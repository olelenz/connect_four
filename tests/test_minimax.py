import pytest

from agents.agent_minimax.minimax import handle_empty_moves_eval, START_VALUE, get_possible_moves_iterative
from agents.game_utils import *

EMPTY_BOARD: int = 0b0000000_0000000_0000000_0000000_0000000_0000000_0000000
DRAW_PLAYER_ONE: int = 0b0001010_0010101_0111011_0101110_0000100_0010100_0100011
DRAW_PLAYER_TWO: int = 0b0110101_0101010_0000100_0010001_0111011_0101011_0011100
DIAGONAL_BOARD_LEFT_TOP: int = 0b0000000_0000000_0000001_0000010_0000100_0001000_0000000

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
