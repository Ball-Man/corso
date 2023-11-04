"""Produce results of several round-robin tournaments."""
import sys
from typing import Any
from itertools import combinations
from functools import partial
from random import Random

import numpy as np

from corso.evaluation import evaluate
from corso.minmax import MinMaxPlayer
from corso.model import RandomPlayer, Corso
from corso.az import AZConvNetwork, AZPlayer

NON_NEURAL_AGENTS_FACTORIES = (
    RandomPlayer,
    partial(MinMaxPlayer, 1, temperature=1e-7),
    partial(MinMaxPlayer, 2, temperature=1e-7),
    partial(MinMaxPlayer, 3, temperature=1e-7),
    partial(MinMaxPlayer, 4, temperature=1e-7),
    partial(MinMaxPlayer, 6, temperature=1e-7),
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
    az_agents_info = sys.argv[2:]
    num_tournaments = 10
    starting_state = Corso()

    rng = Random(seed)

    # Expect pairs of type: (az_network_directory, num_simulations)
    az_agents_dirs = az_agents_info[0::2]
    az_agents_simulations = map(int, az_agents_info[1::2])

    az_policies = tuple(map(AZConvNetwork.load, az_agents_dirs))
    az_agents = list(map(lambda p: AZPlayer(p[0], p[1], 0., seed=seed),
                         zip(az_policies, az_agents_simulations)))

    all_agents = az_agents + [agent(rng=Random(seed)) for agent
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
