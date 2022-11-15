import numpy as np
import random as rd
from typing import Tuple, Optional

from agents.game_utils import BoardPiece, PlayerAction, get_possible_moves
from agents.saved_state import SavedState


def generate_move_random(
        board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Choose a valid, non-full column randomly and return it as `action`

    Parameters
    ----------
    board: numpy.ndarray
        Board to generate the move from.

    player: BoardPiece
        Player to make the move.

    saved_state: Optional[SavedState]
        Can be used to save a state of the game for future calculations. Not used here.

    Returns
    -------
    :Tuple[PlayerAction, Optional[SavedState]]
        Tuple containing the move to play and the saved state.

    """
    return PlayerAction(rd.choices(get_possible_moves(board))[0]), saved_state
