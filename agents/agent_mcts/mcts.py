from typing import Optional, Tuple, List
import numpy as np
import time
import random as rd

from agents.agent_minimax.mcts_tree import MctsTree
from agents.game_utils import PlayerAction, BoardPiece, get_possible_moves, apply_player_action, PLAYER1, \
    PLAYER2, check_end_state, GameState, pretty_print_board, connected_four
from agents.saved_state import SavedState


def generate_move_mcts(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState], seconds: int = 1) -> \
        Tuple[
            PlayerAction, Optional[SavedState]]:
    start_time = time.time()

    game_tree: MctsTree = MctsTree(None, board, None, player)
    if saved_state is not None:
        board_string = pretty_print_board(board)
        children: List[MctsTree] = saved_state.get_tree().get_child_trees()
        for child in children:
            if board_string == pretty_print_board(child.get_board()):
                game_tree = child
                print("I found recent information!: "+str(child.get_last_move()))
                break
    ite: int = 0

    while time.time() - start_time < seconds:
    #while ite < 1500:
        ite = ite+1
        current_node = selection(game_tree)  # is a leaf-node
        #  either direct rollout or add possible moves as children

        if current_node.get_n() != 0:
            #  add possible moves as children, set current node to first children -> expansion
            expansion(current_node)
            current_children = current_node.get_child_trees()
            if len(current_children) != 0:
                current_node = current_children[0]  # check case where there are no more possible moves? -> or is that even needed?
        #  simulation
        result: int = simulation(current_node.get_board(), player, current_node.get_player())
        backpropagation(current_node, result)
        pass

    children: List[MctsTree] = game_tree.get_child_trees()
    if len(children) == 0:  # in case of only one iteration
        raise Exception
    result_tree: MctsTree = children[0]
    for i in range(1, len(children)):
        if children[i].get_t() > result_tree.get_t():
            result_tree = children[i]

    print("hello: " + str(result_tree.get_last_move()))
    return PlayerAction(result_tree.get_last_move()), SavedState(result_tree)


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
    for move in get_possible_moves(current_board):
        leaf_node.add_child_tree(MctsTree(leaf_node, apply_player_action(current_board, move, leaf_node.get_player()), move, BoardPiece(3-leaf_node.get_player())))


def simulation(board: np.ndarray, initial_player: BoardPiece, next_player: BoardPiece) -> int:
    moves: List[PlayerAction] = get_possible_moves(board)
    #while check_end_state(board, other_player) == GameState.STILL_PLAYING:
    while len(moves) != 0:
        board = apply_player_action(board, rd.choice(get_possible_moves(board)), next_player)
        moves = get_possible_moves(board)
        next_player = BoardPiece(3-next_player)

    end: GameState = check_end_state(board, next_player)
    if end == GameState.IS_DRAW:
        return 0
    if next_player == initial_player:
        return -1
    return 1


def backpropagation(current_node: MctsTree, result: int):
    while current_node is not None:
        current_node.increment_n()
        current_node.update_t(result)
        current_node = current_node.get_parent_tree()
