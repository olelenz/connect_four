from typing import List
import numpy as np

from agents.game_utils import BoardPiece, PlayerAction


class MctsTree:
    """
    Class representing a tree used for the mcts algorithm.

    Methods
    ----------
    __init__(self, parent_tree: 'MctsTree', board: np.ndarray, last_move: PlayerAction, player: BoardPiece)
        Constructor of MctsTree class.

    get_parent_tree(self)
        Getter method for self.parent_tree.

    get_board(self)
        Getter method for self.board.

    get_last_move(self)
        Getter method for self.last_move.

    get_player(self)
        Getter method for self.player.

    get_w(self)
        Getter method for self.w.

    get_n(self)
        Getter method for self.n.

    get_child_trees(self)
        Getter method for self.child_trees.

    calculate_ucb(self)
        Method to calculate current ucb value.

    increment_n(self)
        Increments self.n by 1.

    update_w(self, add_to_sum: int)
        Adds input to self.w.

    add_child_tree(self, children: 'MctsTree')
        Adds a child tree to self.child_trees.
    """
    def __init__(self, parent_tree: 'MctsTree', board: np.ndarray, last_move: PlayerAction, player: BoardPiece):
        """
        Constructor of MctsTree class.

        Parameters
        ----------
        parent_tree: MctsTree
            Parent tree.

        board: numpy.ndarray
            Current board position.

        last_move: PlayerAction
            Last move played to reach the current position.

        player: BoardPiece
            The next player to make a move.
        """
        self.parent_tree: 'MctsTree' = parent_tree
        self.board: np.ndarray = board
        self.last_move: PlayerAction = last_move
        self.player: BoardPiece = player

        self.w = 0
        self.n = 0
        self.child_trees: List['MctsTree'] = []

    def get_parent_tree(self) -> 'MctsTree':
        """
        Getter method for self.parent_tree.

        Returns
        -------
        :MctsTree
            Parent tree.
        """
        return self.parent_tree

    def get_board(self) -> np.ndarray:
        """
        Getter method for self.board.

        Returns
        -------
        :numpy.ndarray
            Current board position.
        """
        return self.board

    def get_last_move(self) -> PlayerAction:
        """
        Getter method for self.last_move.

        Returns
        -------
        :PlayerAction
            Last move played.
        """
        return self.last_move

    def get_player(self) -> BoardPiece:
        """
        Getter method for self.player.

        Returns
        -------
        :BoardPiece
            Next player.
        """
        return self.player

    def get_w(self) -> int:
        """
        Getter method for self.w.

        Returns
        -------
        :int
            Value of self.w.
        """
        return self.w

    def get_n(self) -> int:
        """
        Getter method for self.n.

        Returns
        -------
        :int
            Value of self.n.
        """
        return self.n

    def get_child_trees(self) -> List['MctsTree']:
        """
        Getter method for self.child_trees.

        Returns
        -------
        :List[MctsTree]
            List containing all children.
        """
        return self.child_trees

    def calculate_ucb(self) -> float:
        """
        Method to calculate current ucb value.

        Returns
        -------
        :float
            Resulting ucb value.
        """
        if self.n == 0:
            return 1_000_000_000_000
        return (self.w / self.n) + 1 * (np.log(self.parent_tree.n) / self.n) ** (1 / 2)

    def increment_n(self) -> None:
        """
        Increments self.n by 1.
        """
        self.n = self.n + 1

    def update_w(self, add_to_sum: int) -> None:
        """
        Adds input to self.w.

        Parameters
        ----------
        add_to_sum: int
            Value to be added to self.w.
        """
        self.w = self.w + add_to_sum

    def add_child_tree(self, children: 'MctsTree') -> None:  # evaluates type after defining class when executed ->
        # forwarding reference
        """
        Adds a child tree to self.child_trees.

        Parameters
        ----------
        children: MctsTree
            Child tree to be added.
        """
        self.child_trees.append(children)
