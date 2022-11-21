from typing import Optional, Tuple, List
import numpy as np
import time
import random as rd

from agents.agent_mcts.mcts_tree import MctsTree
from agents.game_utils import PlayerAction, BoardPiece, get_possible_moves, apply_player_action, check_end_state, GameState
from agents.saved_state import SavedState


def generate_move_mcts(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState], seconds: int = 4) -> \
        Tuple[PlayerAction, Optional[SavedState]]:
    """
    Function to calculate the next move to play using the mcts algorithm.

    Parameters
    ----------
    board: numpy.ndarray
        Initial board position.

    player: BoardPiece
        The player to make the next move.

    saved_state: Optional[SavedState]
        Can be used to save a state of the game for future calculations or use already saved calculations.

    seconds: int
        Number of seconds allowed for calculation.

    Raises
    ----------
    RuntimeError
        If no move is explored yet and the time for calculation is over.

    Returns
    -------
    :Tuple[PlayerAction, Optional[SavedState]]
        Tuple containing the move to play and the saved state.
    """
    start_time = time.time()

    game_tree: MctsTree = MctsTree(None, board, None, player)
    if saved_state is not None:
        children: List[MctsTree] = saved_state.get_tree().get_child_trees()
        for child in children:
            if np.array_equal(board, child.get_board()):
                game_tree = child
                break

    while time.time() - start_time < seconds:
        current_node = selection(game_tree)  # is a leaf-node
        # either direct rollout or add possible moves as children first
        if current_node.get_n() != 0:
            # add possible moves as children, set current node to first children -> expansion
            expansion(current_node)
            current_children = current_node.get_child_trees()
            if len(current_children) != 0:  # check case where there are no more possible moves
                current_node = current_children[0]
        result: int = simulation(current_node.get_board(), player, current_node.get_player())  # simulation
        backpropagation(current_node, result)

    children: List[MctsTree] = game_tree.get_child_trees()
    if len(children) == 0:  # in case of only one iteration
        raise RuntimeError
    result_tree: MctsTree = children[0]
    for i in range(1, len(children)):  # find best evaluated move
        if children[i].get_w() > result_tree.get_w():
            result_tree = children[i]
    return PlayerAction(result_tree.get_last_move()), SavedState(result_tree)


def selection(current_node: MctsTree) -> MctsTree:
    """
    Function to select the next child node using the ucb value until a leaf node is found.

    Parameters
    ----------
    current_node: MctsTree
        Node to start evaluating from.

    Returns
    -------
    :MctsTree
        The next leaf node to expand/simulate.

    """
    child_trees: List[MctsTree] = current_node.get_child_trees()
    while len(child_trees) != 0:
        final_index: int = 0
        running_index: int = 0
        best_score: float = -1_000_000_000_000
        for child in child_trees:
            ucb: float = child.calculate_ucb()
            if ucb > best_score:
                final_index = running_index
                best_score = ucb
            running_index += 1
        current_node = child_trees[final_index]
        child_trees: List[MctsTree] = current_node.get_child_trees()
    return current_node


def expansion(leaf_node: MctsTree) -> None:
    """
    Function to add all possible moves from the current position as children to a leaf node.

    Parameters
    ----------
    leaf_node: MctsTree
        Current node to be expanded.
    """
    current_board: np.ndarray = leaf_node.get_board()
    for move in get_possible_moves(current_board):
        leaf_node.add_child_tree(MctsTree(leaf_node, apply_player_action(current_board, move, leaf_node.get_player()), move, BoardPiece(3-leaf_node.get_player())))


def simulation(board: np.ndarray, initial_player: BoardPiece, next_player: BoardPiece) -> int:
    """
    Function to simulate a game starting at a starting position until the game ends. Returns 0 for a draw,
    1 if the initial player won, -1 otherwise.

    Parameters
    ----------
    board: numpy.ndarray
        Board to start the simulation from.

    initial_player: BoardPiece
        The player to make the move from the initial position which is  to be evaluated by the mcts algorithm.

    next_player: BoardPiece
        The player to make the next move from the current position.

    Returns
    -------

    """
    moves: List[PlayerAction] = get_possible_moves(board)
    while len(moves) != 0:  # is replaced in the loop
        board = apply_player_action(board, rd.choice(get_possible_moves(board)), next_player)
        moves = get_possible_moves(board)
        next_player = BoardPiece(3-next_player)

    end: GameState = check_end_state(board, next_player)
    if end == GameState.IS_DRAW:
        return 0
    if next_player == initial_player:  # flipped because the player is changed after the last move
        return -1
    return 1


def backpropagation(current_node: MctsTree, result: int) -> None:
    """
    Function to backpropagate the ucb value from the leaf node up to the root node.

    Parameters
    ----------
    current_node: MctsTree
        Current node to start backpropagation from.

    result: int
        Ucb result to be backpropagated.

    """
    while current_node is not None:
        current_node.increment_n()
        current_node.update_w(result)
        current_node = current_node.get_parent_tree()
