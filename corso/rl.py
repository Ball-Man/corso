"""Reinforcement learning approaches for the game."""
import random
from functools import lru_cache
from collections import deque

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from corso.model import (Corso, CellState, Action,
                         DEFAULT_BOARD_SIZE, DEFAULT_PLAYER_NUM)


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
        self.input_dropout = nn.Dropout(0.2)
        self.dense1 = nn.Linear(board_w * board_h * player_num * 2 + 1, 64)
        self.dropout1 = nn.Dropout(0.2)
        self.dense2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.2)
        self.output = nn.Linear(32, board_w * board_h)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # batch = batch.flatten(1)
        batch = self.input_dropout(batch)

        batch = F.relu(self.dense1(batch))
        batch = self.dropout1(batch)

        batch = F.relu(self.dense2(batch))
        batch = self.dropout2(batch)

        return F.log_softmax(self.output(batch))

    def sample_action(self, state: Corso) -> Action:
        """Sample action from policy."""
        policy = self(model_tensor(state).unsqueeze(0))[0]

        # Not all moves given by the network are legal. Mask illegal
        # moves with 0 probabilities.
        policy_mask = torch.zeros_like(policy).view(state.height,
                                                    state.width)
        for action in state._iter_actions():
            policy_mask[action[1:]] = 1

        # TODO: reproducibility
        # random.choices is 3+ times faster than np.random.choice in
        # this context.
        action_index, = random.choices(
            _action_indeces(state.width, state.height),
            policy_mask.flatten() * policy.exp())

        row, column = divmod(action_index, state.width)
        return Action(state.player_index, row, column)
