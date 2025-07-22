import time
import csv
import pandas as pd
import matplotlib.pyplot as plt
from board import Board
from heuristics import SimpleHeuristic
from players import MinMaxPlayer, AlphaBetaPlayer

def measure_runtime_and_evals(player_class, board_size, game_n, depth, num_trials=5):
    width, height = board_size
    total_time = 0.0
    total_evals = 0

    #i want to measure how much it takes on average for the algorithm to settle on a move
    for _ in range(num_trials):
        heuristic = SimpleHeuristic(game_n)
        board = Board(width, height)
        player = player_class(1, game_n, depth, heuristic)

        start = time.time()
        player.make_move(board)
        end = time.time()

        total_time += (end - start)
        #here im using a function from the players.py section to see how many evalutations are made by the algorithm in use
        total_evals += player.get_eval_count()

    #averaging stuff
    avg_time = total_time / num_trials
    avg_evals = total_evals / num_trials
    return avg_time, avg_evals



board_sizes = [(6, 6), (7, 6), (8, 7)]
game_ns = [3, 4]
depths = [2, 3, 4, 5, 6]

with open("benchmark_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Board Size", "Game N", "Depth",
        "MinMax Time (s)", "AlphaBeta Time (s)",
        "Speedup", "MinMax Evals", "AlphaBeta Evals"
    ])

    for board_size in board_sizes:
        for game_n in game_ns:
            for depth in depths:
                t_minmax, eval_minmax = measure_runtime_and_evals(MinMaxPlayer, board_size, game_n, depth)
                t_alphabeta, eval_alphabeta = measure_runtime_and_evals(AlphaBetaPlayer, board_size, game_n, depth)

                speedup = t_minmax / t_alphabeta if t_alphabeta > 0 else float("inf")

                writer.writerow([
                    board_size, game_n, depth,
                    round(t_minmax, 6), round(t_alphabeta, 6),
                    round(speedup, 2),
                    int(eval_minmax), int(eval_alphabeta)
                ])


df = pd.read_csv("benchmark_results.csv")
df["Board Size"] = df["Board Size"].apply(eval)   #converting the sstrings into acutal touples or wont see them

#I only want to make graph of 7,6 board size and N=4, so I'll filter for that
filtered = df[(df["Board Size"] == (7, 6)) & (df["Game N"] == 4)]

plt.figure()
plt.plot(filtered["Depth"], filtered["Speedup"], marker='o', color='tab:blue')
plt.xlabel("Search Depth")
plt.ylabel("Speedup (MinMax / AlphaBeta)")
plt.title("AlphaBeta Speedup vs Depth\nBoard: 7x6, N=4")
plt.grid(True)
plt.tight_layout()
plt.show()

x = filtered["Depth"]
width = 0.35

plt.figure()
plt.bar(x - width/2, filtered["MinMax Evals"], width, label="MinMax")
plt.bar(x + width/2, filtered["AlphaBeta Evals"], width, label="AlphaBeta")
plt.xlabel("Search Depth")
plt.ylabel("Evaluation Count")
plt.title("Board States Evaluated vs Depth\nBoard: 7x6, N=4")
plt.xticks(x)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(filtered["Depth"], filtered["MinMax Time (s)"], marker='o', label="MinMax", color='tab:red')
plt.plot(filtered["Depth"], filtered["AlphaBeta Time (s)"], marker='x', label="AlphaBeta", color='tab:green')
plt.xlabel("Search Depth")
plt.ylabel("Time (seconds)")
plt.title("Runtime vs Depth\nBoard: 7x6, N=4")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
