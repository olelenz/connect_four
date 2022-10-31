import numpy as np
import random as rd
from typing import Tuple, Optional

from agents.game_utils import BoardPiece, SavedState, PlayerAction, NO_PLAYER


def generate_move_random(
        board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    # Choose a valid, non-full column randomly and return it as `action`
    choices: [int] = [0, 1, 2, 3, 4, 5, 6]
    action: int = -1
    while len(choices) != 0:
        action = rd.choice(choices)
        if board[5][action] != NO_PLAYER:
            choices.remove(action)
            action = -1
            continue
        break
    return PlayerAction(action), saved_state
