from __future__ import annotations
from abc import abstractmethod
import numpy as np
import random
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from heuristics import Heuristic
    from board import Board
from tree import Node, build_tree


class PlayerController:
    """Abstract class defining a player
    """
    def __init__(self, player_id: int, game_n: int, heuristic: Heuristic) -> None:
        """
        Args:
            player_id (int): id of a player, can take values 1 or 2 (0 = empty)
            game_n (int): n in a row required to win
            heuristic (Heuristic): heuristic used by the player
        """
        self.player_id = player_id
        self.game_n = game_n
        self.heuristic = heuristic


    def get_eval_count(self) -> int:
        """
        Returns:
            int: The amount of times the heuristic was used to evaluate a board state
        """
        return self.heuristic.eval_count
    

    def __str__(self) -> str:
        """
        Returns:
            str: representation for representing the player on the board
        """
        if self.player_id == 1:
            return 'X'
        return 'O'
        

    @abstractmethod
    def make_move(self, board: Board) -> int:
        """Gets the column for the player to play in

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """
        pass


class MinMaxPlayer(PlayerController):
    def __init__(self, player_id: int, game_n: int, depth: int, heuristic: Heuristic) -> None:

        super().__init__(player_id, game_n, heuristic)
        self.depth: int = depth
        self.nodes_evaluated = 0 #counter


    def make_move(self, board: Board) -> int:
        self.nodes_evaluated = 0

        if self.player_id == 1:
            root_parent = 2
        else:
            root_parent = 1


        root_node = Node(
            board=board,
            player=root_parent,
            move=None,
            depth=self.depth,
            game_n=self.game_n
        )

        build_tree(root_node)

        def minimax(node: Node) -> float:
            from app import winning

            next_player = 1 if node.player == 2 else 2

            state_arr = node.board.get_board_state()
            winner = winning(state_arr, node.game_n)  # 1, 2, -1 (draw), or 0 (not over)

            if winner != 0:
                if winner == self.player_id:
                    node.value = float("+inf")
                elif winner == -1:
                    node.value = 0.0
                else:
                    node.value = float("-inf")
                return node.value

            if node.depth == 0:
                hval = self.heuristic.evaluate_board(self.player_id, node.board)
                self.nodes_evaluated += 1
                node.value = float(hval)
                return node.value


            if next_player == self.player_id:
                best_value = float("-inf")
                for child in node.children:
                    child_val = minimax(child)
                    if child_val > best_value:
                        best_value = child_val
                node.value = best_value
            else:
                best_value = float("+inf")
                for child in node.children:
                    child_val = minimax(child)
                    if child_val < best_value:
                        best_value = child_val
                node.value = best_value

            return node.value


        root_value = minimax(root_node)
        best_score = float("-inf")
        best_move = None
        for child in root_node.children:
            if child.value is not None and child.value > best_score:
                best_score = child.value
                best_move = child.move

        if best_move is None:
            print("[MinMaxPlayer] No valid children found, falling back to heuristic.")
            return self.heuristic.get_best_action(self.player_id, board)

        print(f"[MinMaxPlayer] → Chosen move = {best_move} (score={best_score})\n")
        print(self.nodes_evaluated)
        return best_move


    

class AlphaBetaPlayer(PlayerController):
    def __init__(self, player_id: int, game_n: int, depth: int, heuristic: Heuristic) -> None:

        super().__init__(player_id, game_n, heuristic)
        self.depth: int = depth
        self.nodes_evaluated = 0


    def make_move(self, board: Board) -> int:
        self.nodes_evaluated = 0

        if self.player_id == 1:
            root_parent = 2
        else:
            root_parent = 1

        root_node = Node(
            board=board,
            player=root_parent,
            move=None,
            depth=self.depth,
            game_n=self.game_n
        )

        build_tree(root_node)

        def alphabeta(node: Node, alpha: float, beta: float, maximizing_player: bool) -> float:
            from app import winning

            next_player = 1 if node.player == 2 else 2
            state_arr = node.board.get_board_state()
            winner = winning(state_arr, node.game_n)

            if winner != 0:
                if winner == self.player_id:
                    node.value = float("+inf")
                elif winner == -1:
                    node.value = 0.0
                else:
                    node.value = float("-inf")
                return node.value

            if node.depth == 0:
                hval = self.heuristic.evaluate_board(self.player_id, node.board)
                self.nodes_evaluated += 1
                node.value = float(hval)
                return node.value

            if maximizing_player:
                max_eval = float("-inf")
                for child in node.children:
                    eval = alphabeta(child, alpha, beta, False)
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
                node.value = max_eval
                return max_eval
            else:
                min_eval = float("+inf")
                for child in node.children:
                    eval = alphabeta(child, alpha, beta, True)
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
                node.value = min_eval
                return min_eval

        alphabeta(root_node, float("-inf"), float("+inf"), True)

        best_score = float("-inf")
        best_move = None
        for child in root_node.children:
            if child.value is not None and child.value > best_score:
                best_score = child.value
                best_move = child.move

        if best_move is None:
            print("[AlphaBetaPlayer] No valid move found, falling back to heuristic.")
            return self.heuristic.get_best_action(self.player_id, board)

        print(f"[AlphaBetaPlayer] Chosen move = {best_move} (score = {best_score})\n")
        print(self.nodes_evaluated)
        return best_move


class MonteCarloPlayer(PlayerController):
    def __init__(self, player_id: int, game_n: int, num_playouts: int, heuristic=None) -> None:
        super().__init__(player_id, game_n, heuristic)
        self.num_playouts = num_playouts

    def make_move(self, board: "Board") -> int:
        valid_cols = [c for c in range(board.width) if board.is_valid(c)]
        if not valid_cols:
            # No valid mvoes :(
            return 0

        win_counts = {c: 0 for c in valid_cols}

        for col in valid_cols:
            for _ in range(self.num_playouts):
                winner = self._simulate_random_game(board, col)
                if winner == self.player_id:
                    win_counts[col] += 1

        best_col = max(valid_cols, key=lambda c: (win_counts[c], -c))
        return best_col

    def _simulate_random_game(self, root_board: "Board", root_col: int) -> int:
        import random
        board_copy = root_board.get_new_board(root_col, self.player_id)


        current_player = 1 if self.player_id == 2 else 2
        from app import winning as app_winning  #

        while True:
            state_arr = board_copy.get_board_state()
            winner = app_winning(state_arr, self.game_n)
            if winner != 0:
                return winner

            valid = [c for c in range(board_copy.width) if board_copy.is_valid(c)]
            if not valid:
                return -1

            pick = random.choice(valid)
            board_copy.play(pick, current_player)

            # sswitch player
            current_player = 1 if current_player == 2 else 2



class HumanPlayer(PlayerController):
    """Class for the human player
    Inherits from Playercontroller
    """
    def __init__(self, player_id: int, game_n: int, heuristic: Heuristic) -> None:
        """
        Args:
            player_id (int): id of a player, can take values 1 or 2 (0 = empty)
            game_n (int): n in a row required to win
            heuristic (Heuristic): heuristic used by the player
        """
        super().__init__(player_id, game_n, heuristic)

    
    def make_move(self, board: Board) -> int:
        """Gets the column for the player to play in

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """
        print(board)

        if self.heuristic is not None:
            print(f'Heuristic {self.heuristic} calculated the best move is:', end=' ')
            print(self.heuristic.get_best_action(self.player_id, board) + 1, end='\n\n')

        col: int = self.ask_input(board)

        print(f'Selected column: {col}')
        return col - 1
    

    def ask_input(self, board: Board) -> int:
        """Gets the input from the user

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """
        try:
            col: int = int(input(f'Player {self}\nWhich column would you like to play in?\n'))
            assert 0 < col <= board.width
            assert board.is_valid(col - 1)
            return col
        except ValueError: # If the input can't be converted to an integer
            print('Please enter a number that corresponds to a column.', end='\n\n')
            return self.ask_input(board)
        except AssertionError: # If the input matches a full or non-existing column
            print('Please enter a valid column.\nThis column is either full or doesn\'t exist!', end='\n\n')
            return self.ask_input(board)


#im adding a random player so i have a useful baseline to compare how the algorithms do

class RandomPlayer(PlayerController):
    def __init__(self, player_id: int, game_n: int, heuristic=None):
        super().__init__(player_id, game_n, heuristic)

    def make_move(self, board: "Board") -> int:
        valid_moves = [col for col in range(board.width) if board.is_valid(col)]
        return random.choice(valid_moves)