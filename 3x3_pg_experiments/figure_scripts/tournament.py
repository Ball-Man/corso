"""Produce results of several round-robin tournaments."""
import sys
from typing import Any
from itertools import combinations
from functools import partial
from random import Random

import numpy as np

from corso.evaluation import evaluate
from corso.minmax import MinMaxPlayer
from corso.model import RandomPlayer, Corso, EMPTY_BOARD3X3
from corso.pg import PolicyNetworkPlayer, PolicyNetwork

NON_NEURAL_AGENTS_FACTORIES = (
    RandomPlayer,
    partial(MinMaxPlayer, 1),
    partial(MinMaxPlayer, 2),
    partial(MinMaxPlayer, 3),
    partial(MinMaxPlayer, 10),
)


def get_agents_maps(agents) -> tuple[dict[Any, int], dict[int, Any]]:
    """Return direct and inverse mappings from agent to index.

    Used as indeces in the tournament matrices.
    """
    # Algorithmically terrible, but who cares?
    player_index_map = {agent: agents.index(agent) for agent in agents}
    index_player_map = {index: agent for agent, index
                        in player_index_map.items()}

    return player_index_map, index_player_map


if __name__ == '__main__':
    seed = int(sys.argv[1])
    pg_agents_dirs = sys.argv[2:]
    num_tournaments = 100
    starting_state = Corso(EMPTY_BOARD3X3)

    rng = Random(seed)

    pg_policies = map(PolicyNetwork.load, pg_agents_dirs)
    pg_agents = list(map(PolicyNetworkPlayer, pg_policies))

    all_agents = pg_agents + [agent(rng=Random(seed)) for agent
                              in NON_NEURAL_AGENTS_FACTORIES]
    player_index_map, _ = get_agents_maps(all_agents)
    all_tournaments_results = []

    # TODO?: multiprocessing
    # Play each tournament
    for t in range(num_tournaments):
        print(f'tournament {t + 1}/{num_tournaments}')
        # Keep the number of wins, losses and draws
        tournament_results = np.zeros((len(all_agents), len(all_agents), 3),
                                      dtype=np.int32)

        # Round-robin, shuffle so that who plays as player1 or 2 is
        # randomly decided
        rng.shuffle(all_agents)

        for player1, player2 in combinations(all_agents, 2):
            results_vector = np.array(evaluate(player1, player2,
                                      starting_state=starting_state))
            tournament_results[player_index_map[player1],
                               player_index_map[player2]] += results_vector

        all_tournaments_results.append(tournament_results)

    all_tournaments_matrix = np.stack(all_tournaments_results)

    savefile = f'tournament_{seed}.npy'
    print('saving tournaments matrix of shape', all_tournaments_matrix.shape,
          'at', savefile)
    np.save(f'tournament_{seed}.npy', all_tournaments_matrix)
