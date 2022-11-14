import numpy as np
import random as rd
from typing import Tuple, Optional

from agents.game_utils import BoardPiece, PlayerAction, get_possible_moves
from agents.saved_state import SavedState


def generate_move_random(
        board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    # Choose a valid, non-full column randomly and return it as `action`
    action: int = rd.choices(get_possible_moves(board))
    return PlayerAction(action[0]), saved_state
