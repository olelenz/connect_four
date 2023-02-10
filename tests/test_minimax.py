import pytest

from agents.agent_minimax.minimax import handle_empty_moves_eval, START_VALUE
from agents.game_utils import *


def test_handle_empty_moves_eval_draw():
    ret = handle_empty_moves_eval(PLAYER1, GameState.IS_DRAW, 4)
    assert ret == 0


def test_handle_empty_moves_eval_still_playing():
    with pytest.raises(AttributeError):
        handle_empty_moves_eval(PLAYER1, GameState.STILL_PLAYING, 4)


def test_handle_empty_moves_eval_player_one_won():
    ret = handle_empty_moves_eval(PLAYER1, GameState.IS_WIN, 4)
    assert ret == (-START_VALUE * 2 ** 4)


def test_handle_empty_moves_eval_player_one_won():
    ret = handle_empty_moves_eval(PLAYER2, GameState.IS_WIN, 5)
    assert ret == (START_VALUE * 2 ** 5)


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
