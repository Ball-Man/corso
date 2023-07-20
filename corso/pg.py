"""Policy gradient approaches for the game."""
import os.path
import random
import copy
import datetime
from functools import lru_cache
from collections import deque
from typing import Iterable, Optional, Sequence

import torch
import numpy as np
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from corso.utils import SavableModule
from corso.evaluation import AgentEvaluationStrategy
from corso.model import (Corso, CellState, Action, Player,              # NOQA
                         RandomPlayer, DEFAULT_BOARD_SIZE, DEFAULT_PLAYER_NUM,
                         EMPTY_CELL)


@lru_cache()
def _one_hot_cell(cell: CellState) -> tuple[int, int, int, int]:
    """Retrieve one hot vector corresponding to a specific state.

    Tuple values represent respectively::

    - First player dyed cell
    - First player marble
    - Second player dyed cell
    - Second player marble

    An empty cell is a tuple of zeroes.
    """
    player_index = cell.player_index
    one_hot = [0, 0, 0, 0]

    if player_index <= 0:
        return tuple(one_hot)

    one_hot[2 * (player_index - 1) + cell.marble] = 1
    # We can afford reallocating a tuple, it is going to be cached from
    # now on.
    return tuple(one_hot)


def model_tensor(state: Corso) -> torch.Tensor:
    """Retrieve a tensor representing a game state."""
    # This comes with some necessary reallocations before feeding the
    # structure to the tensor constructor. Things are cached where
    # possible. Time cost of this transformation (5x5 board): ~1.5e-5
    # Organizing the Corso state in a way that is more friendly w.r.t.
    # this representation (e.g. with one hot encoded tuples as cell
    # states instead of the abstract CellState class) is the most viable
    # option after this one.

    board_tensor = torch.Tensor(
        tuple(tuple(map(_one_hot_cell, row)) for row in state.board))

    if state.player_index == 2:
        board_tensor = board_tensor[:, :, [2, 3, 0, 1]]

    return board_tensor.flatten()

    # board_tensor = torch.transpose(board_tensor, 1, 2)
    # return torch.transpose(board_tensor, 0, 1)


@lru_cache()
def _action_indeces(width=DEFAULT_BOARD_SIZE,
                    height=DEFAULT_BOARD_SIZE) -> tuple[int, ...]:
    """Retrieve a cached tuple representing a simple range.

    Use as population of all possible moves, then convert the sampled
    index to the proper action.
    """
    return tuple(range(width * height))


class PolicyNetwork(nn.Module, SavableModule):
    """Game policy approximation network.

    Just some dense blocks.
    """

    def __init__(self, board_size=(DEFAULT_BOARD_SIZE, DEFAULT_BOARD_SIZE),
                 hidden_layers: Iterable[int] = (),
                 num_players=DEFAULT_PLAYER_NUM,
                 dropout=0.0):
        super().__init__()

        self.hidden_layers = tuple(hidden_layers)
        self.num_players = num_players
        self.dropout = dropout

        board_w, board_h = board_size
        self.board_size = board_size

        # W * H * players * 2 + 1 is the input size
        # If the number of hidden layers is 0, fall back to this
        # dimentions.
        # hidden_output_size = board_w * board_h * 16
        hidden_output_size = board_w * board_h * num_players * 2

        layers = []

        for hidden_layer_nodes in hidden_layers:
            layers.append(nn.Linear(hidden_output_size,
                                    hidden_layer_nodes))
            layers.append(nn.ReLU())
            hidden_output_size = hidden_layer_nodes

        # Output is a policy of size board_w * board_h
        self.output = nn.Linear(hidden_output_size, board_w * board_h)

        self.layers = nn.ModuleList(layers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for layer in self.layers:
            batch = layer(batch)
            batch = F.dropout(batch, self.dropout, self.training)

        return F.log_softmax(self.output(batch), dim=1)

    def get_masked_policy(self, state: Corso) -> tuple[torch.Tensor,
                                                       torch.Tensor,
                                                       torch.Tensor]:
        """Sample action from policy.

        Return value is a tuple in the form::

        - Tensor representation of the given state
        - Output tensor from the network (log action probabilities)
        - Action policy as a valid density vector. Illegal moves are
            masked to 0 probability.
        """
        state_tensor = model_tensor(state)
        policy = self(state_tensor.unsqueeze(0))[0]

        # Not all moves given by the network are legal. Mask illegal
        # moves with 0 probabilities.
        policy_mask = torch.zeros_like(
            policy, dtype=torch.bool).view(state.height, state.width)
        for action in state._iter_actions():
            policy_mask[action[1:]] = 1
        policy_mask = policy_mask.flatten()

        masked_policy = policy_mask * policy.exp()
        # If underflowing, sample uniformly between legal moves
        if masked_policy[policy_mask].sum() < 1e-12:
            masked_policy[policy_mask] += 1.

        return state_tensor, policy, masked_policy / masked_policy.sum()

    def get_config(self) -> dict:
        """Return a configuration dict: used to save/load the model."""
        return {'board_size': self.board_size,
                'hidden_layers': self.hidden_layers,
                'num_players': self.num_players}


class ValueFunctionNetwork(nn.Module, SavableModule):
    """State value approximation function."""

    def __init__(self, board_size=(DEFAULT_BOARD_SIZE, DEFAULT_BOARD_SIZE),
                 hidden_layers: Iterable[int] = (),
                 num_players=DEFAULT_PLAYER_NUM,
                 dropout=0.0):
        super().__init__()

        self.hidden_layers = tuple(hidden_layers)
        self.num_players = num_players
        self.dropout = dropout

        board_w, board_h = board_size
        self.board_size = board_size

        # W * H * players * 2 + 1 is the input size
        # If the number of hidden layers is 0, fall back to this
        # dimentions.
        # hidden_output_size = board_w * board_h * 16
        hidden_output_size = board_w * board_h * num_players * 2

        layers = []

        for hidden_layer_nodes in hidden_layers:
            layers.append(nn.Linear(hidden_output_size,
                                    hidden_layer_nodes))
            layers.append(nn.ReLU())
            hidden_output_size = hidden_layer_nodes

        # Output is a scalar, the state value
        self.output = nn.Linear(hidden_output_size, 1)

        self.layers = nn.ModuleList(layers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for layer in self.layers:
            batch = layer(batch)
            batch = F.dropout(batch, self.dropout, self.training)

        return F.tanh(self.output(batch))

    def get_config(self) -> dict:
        """Return a configuration dict: used to save/load the model."""
        return {'board_size': self.board_size,
                'hidden_layers': self.hidden_layers,
                'num_players': self.num_players}


def greedy_sample_action(state: Corso,
                         action_policy: torch.Tensor) -> tuple[int, Action]:
    """Given an action policy, return best scoring action."""
    action_index = action_policy.argmax().item()
    row, column = divmod(action_index, state.width)
    return action_index, Action(state.player_index, row, column)


def sample_action(state: Corso,
                  action_policy: torch.Tensor) -> tuple[int, Action]:
    """Given an action policy, return a sampled action accordingly."""
    # TODO: reproducibility
    # random.choices is 3+ times faster than np.random.choice in
    # this context.
    action_index, = random.choices(
        _action_indeces(state.width, state.height),
        action_policy)

    row, column = divmod(action_index, state.width)
    return action_index, Action(state.player_index, row, column)


def episode(policy_net, opponent: Player, starting_state: Corso = Corso(),
            discount=0.9, player2_sampler=torch.distributions.Bernoulli(0.5)
            ) -> tuple[deque[torch.Tensor], deque[torch.Tensor],
                       deque[int], torch.Tensor]:
    """Play a full episode and return trajectory information.

    In order:

    - A deque of state tensors
    - A deque of predicted log policies for each state
    - A deque of action indeces
    - A list of returns (cumulative rewards per state)
    """
    state_tensors = deque()
    logpolicies = deque()
    action_indeces = deque()
    winner = 1

    state = starting_state
    agent_is_second_player = player2_sampler.sample()
    if agent_is_second_player:
        state = state.step(opponent.select_action(state))

    # Iterations: max number of moves in a game of corso is w * h
    # as the longest game would see each player placing a marble
    # without expanding.
    for _ in range(state.width * state.height + 1):
        with torch.no_grad():
            # Retrieve policy from network, mask illegal moves and sample
            state_tensor, logprobs, action_policy = \
                policy_net.get_masked_policy(state)
            action_index, action = sample_action(state, action_policy)

        state_tensors.append(state_tensor)
        logpolicies.append(logprobs)
        action_indeces.append(action_index)

        if action not in state.actions:
            raise ValueError(f'Action {action} is not legal.')

        # Execute action and opponent's action
        state = state.step(action)
        terminal, winner = state.terminal
        if terminal:
            break

        state = state.step(opponent.select_action(state))
        terminal, winner = state.terminal
        if terminal:
            break

    # Assign rewards based on episode result (winner)
    rewards = torch.zeros((len(state_tensors),))

    # Assign reward
    if winner == 1:
        rewards[-1] = 1
    else:
        rewards[-1] = -1

    if agent_is_second_player:
        rewards[-1] *= -1

    # Account for draws
    # if winner == 0:
    #     rewards[-1] = 0.5

    return state_tensors, logpolicies, action_indeces, rewards


def reinforce(
    policy_net, states: torch.Tensor, logpolicies: torch.Tensor,
    action_indeces: np.ndarray, advantage: torch.Tensor,
    policy_optimizer: torch.optim.Optimizer,
        entropy_coefficient: float) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute a reinforce update, given policy and advantage.

    In particular, the reinforce update is computed by gradient descent
    of the following loss:
    :math:`-\log(\pi(a_t|s_t)) A(s_t, a_t)`
    where :math:`A` is the advantage function. The advantage must be
    precomputed and passed as parameter.

    ``logpolicies`` is expected to be the tensor of log predictions
    obtained during data collection. It is unused in this function.

    Mean policy loss and mean entropy are returned.
    """
    policy_optimizer.zero_grad()

    policies_batch = policy_net(states)
    entropy = -(policies_batch * policies_batch.exp()).sum(1)

    probabilities_batch = policies_batch[:, action_indeces]

    loss = (-probabilities_batch * advantage
            - entropy_coefficient * entropy).mean()

    loss.backward()
    policy_optimizer.step()

    return loss, entropy.mean()


def cumulative_rewards_advantage(rewards: Sequence[float],
                                 discount: float,
                                 state_values: Optional[torch.Tensor] = None
                                 ) -> torch.Tensor:
    """Vanilla REINFORCE advantage function.

    Implemented as: :math:`G`, which is the vector of
    cumulative rewards at each timestep, discounted by ``discount``.
    """
    cumulative_rewards = [0] * len(rewards)
    cumulative_rewards[-1] = rewards[-1]

    for i in reversed(range(0, len(cumulative_rewards) - 1)):
        cumulative_rewards[i] = (discount * cumulative_rewards[i + 1]
                                 + rewards[i])

    return torch.tensor(cumulative_rewards)


def baseline_advantage(rewards: Sequence[float], discount: float,
                       state_values: torch.Tensor) -> torch.Tensor:
    """Baseline advantage function.

    Implemented as: :math:`G - V` where :math:`G` is the vector of
    cumulative rewards at each timestep (discounted by ``discount``)
    and :math:`V` is the vector of estimated state values at each
    timestep.
    """
    return (cumulative_rewards_advantage(rewards, discount, state_values)
            - state_values)


def policy_gradient(
    policy_net, value_net, episodes=1000, episodes_per_epoch=64,
    discount=0.9, policy_update_function=reinforce,
    advantage_function=baseline_advantage,
    entropy_coefficient=0.05,
    evaluation_after=1, save_curriculum_after=1,
    curriculum_size=None,
    policy_lr=1e-3, value_function_lr=1e-3,
    player2_probability=0.5,
    starting_state: Corso = Corso(),
    evaluation_strageties: Sequence[AgentEvaluationStrategy] = (),
        writer: Optional[SummaryWriter] = None):
    """ """
    # Build a default writer if not provided
    if writer is None:
        writer = SummaryWriter(
            os.path.join('runs',
                         datetime.datetime.now().strftime(r'%F-%H-%M-%S')))

    optimizer = optim.Adam(policy_net.parameters(), policy_lr)
    value_optimizer = optim.Adam(value_net.parameters(), value_function_lr)

    curriculum = deque([RandomPlayer()], maxlen=curriculum_size)

    player2_sampler = torch.distributions.Bernoulli(player2_probability)

    # Epochs
    for global_episode_index in range(0, episodes, episodes_per_epoch):
        policy_net.train()
        optimizer.zero_grad()

        value_net.train()
        value_optimizer.zero_grad()

        state_tensors = deque()
        logpolicies = deque()
        action_indeces = deque()
        cumulative_rewards = []
        rewards = []

        # Episodes
        opponent = random.choice(curriculum)
        episodes_returns = 0
        for episode_index in range(episodes_per_epoch):
            (ep_state_tensors, ep_logpolicies, ep_action_indeces,
             ep_rewards) = \
                episode(policy_net, opponent, starting_state,
                        discount, player2_sampler=player2_sampler)

            # Stack states episode wise for later retrieval. Keeping
            # the trajectories separate is necessary for advantage
            # calculations, but will eventually be merged together in
            # order to fit the value function estimator
            ep_wise_state_tensor = torch.stack(tuple(ep_state_tensors))

            # Compute cumulative rewards in order to fit
            ep_cumulative_rewards = cumulative_rewards_advantage(
                ep_rewards, discount)

            episodes_returns += ep_cumulative_rewards[-1]       # Analysis

            state_tensors.append(ep_wise_state_tensor)
            logpolicies += ep_logpolicies
            action_indeces += ep_action_indeces
            rewards.append(ep_rewards)
            cumulative_rewards.append(ep_cumulative_rewards)

        writer.add_scalar('train/average_return',
                          episodes_returns / episodes_per_epoch,
                          global_episode_index)

        cumulative_rewards_batch = torch.cat(cumulative_rewards)
        states_batch = torch.cat(tuple(state_tensors))
        logpolicies_batch = torch.stack(tuple(logpolicies))

        assert states_batch.shape[0] == cumulative_rewards_batch.shape[0]
        action_indeces = np.array(action_indeces)

        # Fit value function
        values_batch = value_net(states_batch).squeeze()
        value_loss = F.mse_loss(values_batch, cumulative_rewards_batch)

        value_loss.backward()
        value_optimizer.step()

        # Compute advantage
        # TODO: generalize advantage function through parameter
        # TODO: normalize advantage (?)
        advantage = []
        with torch.no_grad():
            for episode_states, episode_rewards in zip(state_tensors, rewards):
                values_estimates = value_net(episode_states).squeeze()
                advantage.append(
                    advantage_function(episode_rewards, discount,
                                       values_estimates))
        advantage_batch = torch.cat(advantage)

        # Fit policy
        policy_loss, entropy = policy_update_function(
            policy_net, states_batch, logpolicies_batch, action_indeces,
            advantage_batch, optimizer, entropy_coefficient)

        writer.add_scalar('train/entropy', entropy,
                          global_episode_index)
        writer.add_scalar('train/value_loss', value_loss, global_episode_index)
        writer.add_scalar('train/policy_loss', policy_loss,
                          global_episode_index)

        # Evaluation
        epoch, local_episode = divmod(global_episode_index, episodes_per_epoch)
        if local_episode == 0 and ((epoch + 1) % evaluation_after == 0
                                   or epoch == 0):
            policy_net.eval()
            policy_player = PolicyNetworkPlayer(policy_net)

            for evaluation_stragety in evaluation_strageties:
                evaluation_results = evaluation_stragety.evaluate(
                    policy_player, starting_state=starting_state)

                writer.add_scalars(f'eval/{evaluation_stragety.get_name()}',
                                   evaluation_results._asdict(),
                                   global_episode_index)

        # Add current agent to curriculum
        if local_episode == 0 and (epoch + 1) % save_curriculum_after == 0:
            policy_net_copy = copy.deepcopy(policy_net)
            policy_net_copy.eval()
            curriculum.append(
                PolicyNetworkPlayer(policy_net_copy,
                                    sampling_function=sample_action))


# def augmentation_inversion(state_tensors: torch.Tensor,
#                            action_indeces: Iterable[int],
#                            state_width: int,
#                            state_height: int) -> torch.Tensor:
#     """Invert player cells in the given tensors (all of them).

#     ``state_tensors`` is expected to be a stack of state tensors
#     (obtained through :func:`model_tensor`). The stack dimension must
#     be 0 (default behaviour of ``torch.stack``), similarly to a batch.

#     ``action_indeces`` shall be a collection of the indeces of the
#     actions executed for each given state in ```state_tensors` during
#     episode self-play. Some augmentation techniques (e.g. rotation)
#     may require manipulation of the action index to achieve invariance.
#     It is unused when applying inversion.
#     """
#     # Last value is the player index
#     # Dimensions: batch, height, width, cell binary vector (4)
#     planes = state_tensors[:, :-1].view(-1, state_height, state_width, 4)

#     # Invert player planes by swapping the first two values in each
#     # binary vector with the last two
#     planes = planes[:, :, :, [2, 3, 0, 1]]

#     # A new tensor has to be allocated in order to attach the current
#     # player's.
#     return torch.cat((planes.flatten(start_dim=1),
#                       1 - state_tensors[:, -1].unsqueeze(1)),
#                      dim=1)


class PolicyNetworkPlayer(Player):
    """Player whose policy is computed via :class:`PolicyNetwork`."""

    def __init__(self, network: PolicyNetwork, verbose=False,
                 sampling_function=greedy_sample_action):
        self.policy_network = network
        self.verbose = verbose
        self.sampling_function = sampling_function

    def select_action(self, state: Corso) -> Action:
        """ """
        with torch.no_grad():
            _, _, action_policy = self.policy_network.get_masked_policy(state)
            if self.verbose:
                print(action_policy.view(state.height, state.width))

            return self.sampling_function(state, action_policy)[1]
