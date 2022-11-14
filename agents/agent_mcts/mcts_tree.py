from typing import List
import numpy as np

from agents.game_utils import BoardPiece, PlayerAction


class MctsTree:
    def __init__(self, parent_tree, board, last_move, player):
        self.parent_tree: 'MctsTree' = parent_tree
        self.w = 0
        self.n = 0
        self.child_trees: List['MctsTree'] = []
        self.board: np.ndarray = board
        self.last_move: PlayerAction = last_move
        self.player = player

    def get_player(self):
        return self.player

    def get_last_move(self):
        return self.last_move

    def get_parent_tree(self):
        return self.parent_tree

    def get_board(self) -> np.ndarray:
        return self.board

    def get_ucb(self) -> float:
        if self.n == 0:
            return 1_000_000_000_000
        return (self.w / self.n) + 20 * (np.log(self.parent_tree.n) / self.n) ** (1 / 2)

    def increment_n(self) -> None:
        self.n = self.n + 1

    def update_w(self, add_to_sum: int) -> None:
        self.w = self.w + add_to_sum

    def get_w(self) -> int:
        return self.w

    def get_n(self) -> int:
        return self.n

    def add_child_tree(self, children: 'MctsTree'):  # evaluates type after defining class when executed -> forwarding
        # reference
        self.child_trees.append(children)

    def get_child_trees(self) -> List['MctsTree']:
        return self.child_trees

    def get_parent_tree(self):
        return self.parent_tree
