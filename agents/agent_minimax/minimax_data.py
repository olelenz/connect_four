from dataclasses import dataclass

from agents.game_utils import BoardPiece, PlayerAction


@dataclass
class MinimaxCalculation:
    """
    Class for keeping track of the current data for the minimax algorithm.
    """

    board_player_one: int
    board_player_two: int
    current_player: BoardPiece
    next_moves: list[int]
    moves_line: list[int]
    alpha: list[int, [PlayerAction]]
    beta: list[int, [PlayerAction]]
    dictionary: {}
    minmax: int

