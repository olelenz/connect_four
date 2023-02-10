import pytest

from agents.agent_minimax.minimax_data import MinimaxCalculation
from agents.game_utils import PlayerAction


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

    test_data = MinimaxCalculation(0, 0, PlayerAction(1), [1], [2], [1, [2]], [3, [4]], {}, 0)
    print(test_data.__repr__())
    test_data.__setattr__("board_player_one", 30)
    print(test_data.__repr__())
