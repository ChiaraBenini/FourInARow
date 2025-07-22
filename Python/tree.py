# tree.py

from board import Board
from typing import List, Optional


class Node:
    def __init__(
            self,
            board: Board,
            player: int,
            move: Optional[int] = None,
            depth: int = 0,
            game_n: int = 4
    ) -> None:
        self.game_n = game_n
        self.board: Board = board
        self.player: int = player
        self.move: Optional[int] = move
        self.children: List["Node"] = []
        self.depth: int = depth
        self.value: Optional[int] = None  # to store minmax later


def build_tree(root: Node) -> None:
    from app import winning

    state_array = root.board.get_board_state()
    winner = winning(state_array, root.game_n)

    # Stop condition
    if root.depth == 0 or winner != 0:
        return

    # flip turnss
    next_player = 1 if root.player == 2 else 2

    for col in range(root.board.width):
        if root.board.is_valid(col):
            # simulate moves
            new_board = root.board.get_new_board(col, next_player)

            # make node
            child = Node(
                board=new_board,
                player=next_player,
                move=col,
                depth=root.depth - 1
            )

            root.children.append(child)

            #peint deubgss
            """print((
                f"[build_tree] depth={root.depth}  "
                f"parent_move={root.move}  "
                f"child_move={col}  "
                f"player={next_player}"
            ))"""

            build_tree(child)



