"""Reinforcement learning approaches for the game."""
import random
from functools import lru_cache
from itertools import cycle
from collections import deque

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
                 player_num=DEFAULT_PLAYER_NUM):
        super().__init__()

        board_w, board_h = board_size
        self.dense1 = nn.Linear(board_w * board_h * player_num * 2 + 1, 32)
        self.dense2 = nn.Linear(32, 16)
        self.output = nn.Linear(16, board_w * board_h)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # batch = batch.flatten(1)
        batch = F.relu(self.dense1(batch))
        batch = F.relu(self.dense2(batch))

        return F.log_softmax(self.output(batch), dim=1)

    def get_masked_policy(self, state: Corso) -> tuple[torch.Tensor,
                                                       torch.Tensor]:
        """Sample action from policy.

        Return value is a tuple in the form::

        - Output tensor from the network (log action probabilities)
        - Action policy as a valid density vector. Illegal moves are
            masked to 0 probability.
        """
        policy = self(model_tensor(state).unsqueeze(0))[0]

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

        return policy, masked_policy / masked_policy.sum()


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


def reinforce(episodes=1000, discount=0.9):
    """ """
    policy_net = PolicyNetwork((3, 3))
    optimizer = optim.Adam(policy_net.parameters(), 0.001)

    loss_history = deque()
    evaluation_history = deque()

    for episode in range(episodes):            # Episodes
        optimizer.zero_grad()
        policy_net.train()

        probability_tensors = deque()
        result = 0

        state = Corso(BOARD3X3)
        # Iterations: max number of moves in a game of corso is w * h
        # as the longest game would see each player placing a marble
        # without expanding.
        for _ in range(state.width * state.height):
            # Retrieve policy from network, mask illegal moves and sample
            logprobs, action_policy = policy_net.get_masked_policy(state)
            action_index, action = sample_action(state, action_policy)

            probability_tensors.append(logprobs[action_index])

            if action not in state.actions:
                raise ValueError(f'Action {action} is not legal.')

            state = state.step(action)
            terminal, winner = state.terminal
            if terminal:
                print(f'Ending episode {episode + 1}')
                result = winner - 1
                break

        # Assign rewards based on episode result (winner)
        rewards = torch.zeros((len(probability_tensors),))

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

        probability_batch = torch.stack(tuple(probability_tensors))

        loss = (-cumulative_rewards * probability_batch).mean()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        print('Loss', loss.item())

        # Evaluation
        if episode % 100 == 0:
            policy_net.eval()

            evaluation_results = evaluate(PolicyNetworkPlayer(policy_net),
                                          RandomPlayer(),
                                          n_games=100)
            evaluation_history.append(evaluation_results)
            print('Evaluation results', evaluation_results)

    return loss_history, evaluation_history


def evaluate(player1: Player, player2: Player,
             starting_state=Corso(BOARD3X3),
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
        _, action_policy = self.policy_network.get_masked_policy(state)
        return greedy_sample_action(state, action_policy)[1]
