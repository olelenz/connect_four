from typing import Optional, Tuple
import numpy as np

from agents.game_utils import PlayerAction, BoardPiece, SavedState


def generate_move_mcts(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState], depth: int = 4) -> Tuple[
    PlayerAction, Optional[SavedState]]:
    return PlayerAction(0), None
