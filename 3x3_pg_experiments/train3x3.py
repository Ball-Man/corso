# from timeit import timeit
import random
import sys
from functools import partial

import torch
from torch.utils.tensorboard import SummaryWriter

from corso.model import Corso, RandomPlayer, EMPTY_BOARD3X3     # NOQA
from corso.evaluation import AgentEvaluationStrategy
from corso.minmax import MinMaxPlayer
from corso.pg import (PolicyNetwork, ValueFunctionNetwork,      # NOQA
                      policy_gradient, reinforce, ppo_clip,
                      td_advantage, montecarlo_values, bootstrapped_values,
                      cumulative_rewards_advantage, baseline_advantage)

torch.use_deterministic_algorithms(True)

if len(sys.argv) > 1:
    seed = int(sys.argv[1])
else:
    seed = None

episodes = 2 ** 19
starting_state = Corso(EMPTY_BOARD3X3)        # EMPTY_BOARD3X3

# Initialize in a reproducible way
with torch.random.fork_rng():
    if seed is not None:
        torch.default_generator.manual_seed(seed)

    policy_net = PolicyNetwork((starting_state.width, starting_state.height),
                               hidden_layers=[36], dropout=0.)
    value_net = ValueFunctionNetwork((starting_state.width,
                                      starting_state.height),
                                     hidden_layers=[36], dropout=0.)

evaluation_strategies = (
    AgentEvaluationStrategy(partial(RandomPlayer, rng=random.Random(seed)),
                            100, name='vs_random'),
    AgentEvaluationStrategy(partial(MinMaxPlayer, 1, rng=random.Random(seed)),
                            50, name='vs_minmax1'),
    AgentEvaluationStrategy(partial(MinMaxPlayer, 2, rng=random.Random(seed)),
                            50, name='vs_minmax2'),
    AgentEvaluationStrategy(partial(MinMaxPlayer, rng=random.Random(seed)),
                            30, name='vs_minmax3'),
)

if __name__ == '__main__':
    print('training with seed', seed)
    writer = SummaryWriter(f'runs/3x3_{seed}')

    policy_gradient(policy_net, value_net,
                    policy_update_function=ppo_clip,
                    advantage_function=td_advantage,
                    evaluation_strageties=evaluation_strategies,
                    value_function_target=bootstrapped_values,
                    episodes=episodes,
                    policy_lr=1e-4,
                    value_function_lr=1e-4,
                    discount=1, episodes_per_epoch=16,
                    starting_state=starting_state,
                    evaluation_after=30,
                    entropy_coefficient=0.1,
                    save_curriculum_after=20,
                    curriculum_size=50,
                    value_function_minibatches=4,
                    player2_probability=0.5,
                    value_functon_weight_decay=0.,
                    policy_weight_decay=0.,
                    seed=seed,
                    writer=writer
                    )

    policy_net.save(f'policy_net_{seed}')
    value_net.save(f'value_net_{seed}')
