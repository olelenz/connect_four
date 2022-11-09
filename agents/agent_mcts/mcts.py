from typing import Optional, Tuple, List
import numpy as np
import time
import random as rd

from agents.agent_minimax.mcts_tree import MctsTree
from agents.game_utils import PlayerAction, BoardPiece, SavedState, get_possible_moves, apply_player_action, PLAYER1, \
    PLAYER2, check_end_state, GameState, pretty_print_board, connected_four


def generate_move_mcts(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState], seconds: int = 5) -> \
        Tuple[
            PlayerAction, Optional[SavedState]]:
    start_time = time.time()

    game_tree: MctsTree = MctsTree(None, board, player, None)
    current_node: MctsTree = game_tree

    while time.time() - start_time < seconds:
        current_node = selection(current_node)  # is a leaf-node
        #  either direct rollout or add possible moves as children

        if current_node.get_n() != 0:
            #  add possible moves as children, set current node to first children -> expansion
            expansion(current_node)
            current_children = current_node.get_child_trees()
            if len(current_children) != 0:
                current_node = current_children[0]  # check case where there are no more possible moves? -> or is that even needed?
        #  simulation
        result: int = simulation(current_node.get_board(), player)
        backpropagation(current_node, result)

    children: List[MctsTree] = game_tree.get_child_trees()
    result_tree: MctsTree = children[0]
    for i in range(1, len(children)):
        if children[i].get_t() > result_tree.get_t():
            result_tree = children[i]

    print("hello: " + str(result_tree.get_last_move()))
    return PlayerAction(result_tree.get_last_move()), None


def selection(current_node: MctsTree) -> MctsTree:
    child_trees: List[MctsTree] = current_node.get_child_trees()
    while len(child_trees) != 0:
        final_index: int = 0
        running_index: int = 0
        best_score: float = -1_000_000_000_000
        for child in child_trees:
            ucb: float = child.get_ucb()
            if ucb > best_score:
                final_index = running_index
                best_score = ucb
            running_index += 1
        current_node = child_trees[final_index]
        child_trees: List[MctsTree] = current_node.get_child_trees()
    return current_node


def expansion(leaf_node: MctsTree):
    current_board: np.ndarray = leaf_node.get_board()
    if leaf_node.player == PLAYER1:
        next_player: BoardPiece = PLAYER2
    else:
        next_player: BoardPiece = PLAYER1
    for move in get_possible_moves(current_board):
        leaf_node.add_child_tree(MctsTree(leaf_node, apply_player_action(current_board, move, leaf_node.player),
                                          next_player, move))


def simulation(board: np.ndarray, initial_player: BoardPiece) -> int:
    if initial_player == PLAYER1:
        other_player: BoardPiece = PLAYER2
    else:
        other_player: BoardPiece = PLAYER1
    moves: List[PlayerAction] = get_possible_moves(board)
    #while check_end_state(board, other_player) == GameState.STILL_PLAYING:
    while len(moves) != 0:
        if other_player == PLAYER1:
            other_player: BoardPiece = PLAYER2
        else:
            other_player: BoardPiece = PLAYER1
        board = apply_player_action(board, rd.choice(get_possible_moves(board)), other_player)
        moves = get_possible_moves(board)

    end: GameState = check_end_state(board, other_player)
    if end == GameState.IS_DRAW:
        return 0
    if not connected_four(board, PLAYER1) and not connected_four(board, PLAYER2):
        print(pretty_print_board(board))
        raise Exception  # that should not happen
    if other_player == initial_player:
        return 10
    return -10


def backpropagation(current_node: MctsTree, result: int):
    while current_node is not None:
        current_node.increment_n()
        current_node.update_t(result)
        current_node = current_node.get_parent_tree()
