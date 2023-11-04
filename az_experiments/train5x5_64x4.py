import random
from functools import partial

import corso.az as az
from corso.model import Corso, RandomPlayer
from corso.evaluation import AgentEvaluationStrategy
from corso.minmax import MinMaxPlayer

seed = 1


if __name__ == '__main__':
    evaluation_strategies = (
        AgentEvaluationStrategy(partial(RandomPlayer, rng=random.Random(seed)),
                                50, name='vs_random'),
        AgentEvaluationStrategy(partial(MinMaxPlayer, 1,
                                        rng=random.Random(seed)),
                                30, name='vs_minmax1'),
        AgentEvaluationStrategy(partial(MinMaxPlayer, 2,
                                        rng=random.Random(seed)),
                                30, name='vs_minmax2'),
        AgentEvaluationStrategy(partial(MinMaxPlayer, rng=random.Random(seed)),
                                30, name='vs_minmax3'),
    )

    network = az.AZConvNetwork((5, 5), (64, 64, 64), (32,), 2, (512,),
                               dropout=0.3)

    state = Corso()

    az.alphazero(network=network,
                 iterations=1,
                 episodes=1,
                 simulations=10,
                 epochs_per_iteration=5,
                 starting_state=state,
                 evaluation_strageties=evaluation_strategies,
                 seed=seed)

    network.save('az_network_5x5')
