"""Reinforcement learning approaches for the game."""
import random
from functools import lru_cache
from itertools import cycle
from collections import deque
from typing import Iterable

import torch
from torch import optim
from torch import nn
from torch.nn import functional as F

from corso.model import (Corso, CellState, Action, Player, RandomPlayer,
                         DEFAULT_BOARD_SIZE, DEFAULT_PLAYER_NUM, EMPTY_CELL)

BOARD2X2 = ((EMPTY_CELL, EMPTY_CELL), (EMPTY_CELL, EMPTY_CELL))
BOARD3X3 = ((EMPTY_CELL, EMPTY_CELL, EMPTY_CELL),
            (EMPTY_CELL, EMPTY_CELL, EMPTY_CELL),
            (EMPTY_CELL, EMPTY_CELL, EMPTY_CELL))


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
    return torch.cat((board_tensor.flatten(),
                      torch.Tensor((state.player_index - 1,))))


@lru_cache()
def _action_indeces(width=DEFAULT_BOARD_SIZE,
                    height=DEFAULT_BOARD_SIZE) -> tuple[int, ...]:
    """Retrieve a cached tuple representing a simple range.

    Use as population of all possible moves, then convert the sampled
    index to the proper action.
    """
    return tuple(range(width * height))


class PolicyNetwork(nn.Module):
    """Game policy approximation network.

    Just some dense blocks.
    """

    def __init__(self, board_size=(DEFAULT_BOARD_SIZE, DEFAULT_BOARD_SIZE),
                 first_hidden_size: int = 64,
                 num_hidden_layers: int = 2,
                 player_num=DEFAULT_PLAYER_NUM):
        super().__init__()

        self.first_hidden_size = first_hidden_size
        self.num_hidden_layers = num_hidden_layers

        board_w, board_h = board_size
        self.board_w = board_w
        self.board_h = board_h

        # W * H * players * 2 + 1 is the input size
        # If the number of hidden layers is 0, fall back to this
        # dimentions.
        hidden_output_size = board_w * board_h * player_num * 2 + 1

        layers = []
        for i in range(num_hidden_layers):
            internal_hidden_size = first_hidden_size // 2 ** i
            layers.append(nn.Linear(hidden_output_size,
                                    first_hidden_size // 2 ** i))
            hidden_output_size = internal_hidden_size

        # Output is a policy of size board_w * board_h
        self.output = nn.Linear(hidden_output_size, board_w * board_h)

        self.layers = nn.ModuleList(layers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for layer in self.layers:
            batch = F.relu(layer(batch))

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


def reinforce(episodes=1000, discount=0.9,
              starting_state: Corso = Corso()):
    """ """
    policy_net = PolicyNetwork((starting_state.width, starting_state.height))
    optimizer = optim.Adam(policy_net.parameters(), 0.001)

    loss_history = deque()
    evaluation_history = deque()

    bernoulli = torch.distributions.Bernoulli(0.5)

    for episode in range(episodes):            # Episodes
        optimizer.zero_grad()
        policy_net.train()

        state_tensors = deque()
        action_indeces = deque()
        winner = 1

        state = starting_state
        # Iterations: max number of moves in a game of corso is w * h
        # as the longest game would see each player placing a marble
        # without expanding.
        for _ in range(state.width * state.height):
            with torch.no_grad():
                # Retrieve policy from network, mask illegal moves and sample
                state_tensor, logprobs, action_policy = \
                    policy_net.get_masked_policy(state)
                action_index, action = sample_action(state, action_policy)

            state_tensors.append(state_tensor)
            action_indeces.append(action_index)

            if action not in state.actions:
                raise ValueError(f'Action {action} is not legal.')

            state = state.step(action)
            terminal, winner = state.terminal
            if terminal:
                print(f'Ending episode {episode + 1}')
                # result = winner - 1
                break

        # Assign rewards based on episode result (winner)
        rewards = torch.zeros((len(state_tensors),))

        # The game could finish in an odd number of moves, adjust
        # rewards based on this and on the winner
        if len(rewards) % 2 == 0:
            rewards[-(3 - winner)] = 1
        else:
            rewards[-winner] = 1

        # Account for draws
        # if winner == 0:
        #     rewards[-2:] = torch.tensor([0.5, 0.5])

        # Cumulative rewards
        cumulative_rewards = torch.zeros_like(rewards)
        cumulative_rewards[-2:] = rewards[-2:]      # Prevent 0 reward

        for i in reversed(range(0, len(cumulative_rewards) - 2)):
            # Pick rewards by skipping one action, so that the winner
            # and the loser actions can have separate counts
            cumulative_rewards[i] = (discount * cumulative_rewards[i + 2]
                                     + rewards[i])

        # Recompute policy on the sequence of states and optimize.
        # Recomputation is a potential slowdown w.r.t. using directly
        # the scores computed during training, but allows for data
        # augmentation.
        inversion_map = bernoulli.sample((len(state_tensors),)).bool()
        states_batch = torch.stack(tuple(state_tensors))
        inverted_states = augmentation_inversion(states_batch[inversion_map],
                                                 action_indeces, state.width,
                                                 state.height)
        states_batch[inversion_map] = inverted_states
        policies_batch = policy_net(states_batch)
        probabilities_batch = policies_batch[range(policies_batch.size(0)),
                                             action_indeces]

        loss = (-cumulative_rewards * probabilities_batch).mean()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        print('Loss', loss.item())

        # Evaluation
        if episode % 100 == 0:
            policy_net.eval()

            evaluation_results = evaluate(PolicyNetworkPlayer(policy_net),
                                          RandomPlayer(),
                                          starting_state=starting_state,
                                          n_games=100)
            evaluation_history.append(evaluation_results)
            print('Evaluation results', evaluation_results)

    return loss_history, evaluation_history


def augmentation_inversion(state_tensors: torch.Tensor,
                           action_indeces: Iterable[int],
                           state_width: int,
                           state_height: int) -> torch.Tensor:
    """Invert player cells in the given tensors (all of them).

    ``state_tensors`` is expected to be a stack of state tensors
    (obtained through :func:`model_tensor`). The stack dimension must
    be 0 (default behaviour of ``torch.stack``), similarly to a batch.

    ``action_indeces`` shall be a collection of the indeces of the
    actions executed for each given state in ```state_tensors` during
    episode self-play. Some augmentation techniques (e.g. rotation)
    may require manipulation of the action index to achieve invariance.
    It is unused when applying inversion.
    """
    # Last value is the player index
    # Dimensions: batch, height, width, cell binary vector (4)
    planes = state_tensors[:, :-1].view(-1, state_height, state_width, 4)

    # Invert player planes by swapping the first two values in each
    # binary vector with the last two
    planes = planes[:, :, :, [2, 3, 0, 1]]

    # A new tensor has to be allocated in order to attach the current
    # player's.
    return torch.cat((planes.flatten(start_dim=1),
                      1 - state_tensors[:, -1].unsqueeze(1)),
                     dim=1)


def evaluate(player1: Player, player2: Player,
             starting_state=Corso(),
             n_games=1) -> tuple[int, int, int]:
    """Play automated games and return the results.

    The returned tuple is in the form (draw, p1 wins, p2 wins).
    """
    results = [0, 0, 0]

    for _ in range(n_games):
        state = starting_state
        players = cycle((player1, player2))

        terminal, winner = state.terminal
        while not terminal:
            player = next(players)
            state = state.step(player.select_action(state))

            terminal, winner = state.terminal

        results[winner] += 1

    return tuple(results)


class PolicyNetworkPlayer(Player):
    """Player whose policy is computed via :class:`PolicyNetwork`."""

    def __init__(self, network: PolicyNetwork):
        self.policy_network = network

    def select_action(self, state: Corso) -> Action:
        """ """
        with torch.no_grad():
            _, _, action_policy = self.policy_network.get_masked_policy(state)
            return greedy_sample_action(state, action_policy)[1]
