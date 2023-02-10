from dataclasses import dataclass, field

from agents.game_utils import BoardPiece, PlayerAction

MAX_VALUE: int = 1_000_000_000_000_000_000


@dataclass
class MinimaxCalculation:
    """
    Class for keeping track of the current data for the minimax algorithm.
    """

    depth: int
    board_player_one: int
    board_player_two: int
    current_player: BoardPiece
    dictionary: {}
    alpha: list[int, [PlayerAction]]
    beta: list[int, [PlayerAction]]
    minmax: int = 0  # initialised to maximise
    next_moves: list[int] = field(default_factory=list)
    moves_line: list[int] = field(default_factory=list)

