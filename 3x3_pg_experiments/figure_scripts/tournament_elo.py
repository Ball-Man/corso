"""Given a tournaments tensor, compute Elo of agents.

Given more tournaments in the same format played with different seeds,
provide a confidence interval is plotted as well.
"""
import sys

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd

from tournament import get_agents_maps


AGENTS_NAMES = (
    'REINFORCE',
    'REINFORCE baseline',
    'Actor-Critic',
    'PPO',
    'Random',
    r'MM-$1$',
    r'MM-$2$',
    r'MM-$3$',
    r'MM-$\infty$'
)

STARTING_ELO = 1000


def get_elo_deltas(winner_elo, loser_elo, k=30) -> tuple[float, float]:
    """Compute elo updates of a game.

    Draws are not considered, updates are returned in the form:
    ``(winner_update, loser_update)``.
    """
    q_winner = 10 ** (winner_elo / 400)
    q_loser = 10 ** (loser_elo / 400)

    expected_winner = q_winner / (q_winner + q_loser)
    expected_loser = 1 - expected_winner

    return k * (1 - expected_winner), -k * expected_loser


def get_final_elos(tournament_tensor) -> np.ndarray:
    """Given a tensor having results of multiple games, get player elos."""
    elos = np.full((len(AGENTS_NAMES),), STARTING_ELO)

    # for tournament_tensor in tournament_tensors:
    # tournament_tensor = tournament_tensors

    for tournament in range(tournament_tensor.shape[0]):
        # Store deltas for the duration of the tournament, then update
        elos_deltas = np.zeros_like(elos)

        for row in range(tournament_tensor.shape[1]):
            for column in range(tournament_tensor.shape[2]):
                # Skip if no game was played (ignore draws)
                if not np.any(tournament_tensor[tournament, row, column, 1:]):
                    continue

                game_elos = elos[row], elos[column]

                reverse_player = tournament_tensor[tournament, row, column, 2]
                if reverse_player:
                    deltas = reversed(get_elo_deltas(*reversed(game_elos)))
                else:
                    deltas = get_elo_deltas(*game_elos)

                row_elo_delta, column_elo_delta = deltas
                elos_deltas[row] += row_elo_delta
                elos_deltas[column] += column_elo_delta

        # Update global Elo
        elos += elos_deltas

    return elos


if __name__ == '__main__':
    tournaments_tensors_files = sys.argv[1:]
    tournament_tensors = tuple(map(np.load, tournaments_tensors_files))

    _, index_player_map = get_agents_maps(AGENTS_NAMES)

    elos = np.stack(tuple(map(get_final_elos, tournament_tensors)))
    elos_df = pd.DataFrame(elos, columns=AGENTS_NAMES)      # Add names

    # Plot
    plt.figure(figsize=(9, 7))

    sns.barplot(elos_df, color='#597dbf')

    plt.ylabel('Elo score (start=1000, k=32)')
    plt.xticks(rotation=90)
    plt.grid(axis='y')
    plt.tight_layout()

    plt.savefig('tournament_elo.svg', format='svg')
