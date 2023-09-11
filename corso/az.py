"""AlphaZero-like approaches to solve the game.

Current code is designed for two player games.
"""
from typing import Iterable, Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from corso.model import Corso, Action, DEFAULT_BOARD_SIZE, DEFAULT_PLAYER_NUM
from corso.utils import SavableModule, bitmap_cell


class PriorPredictorProtocol(Protocol):
    """Protocol for state predictors (neural networks) for AZ."""

    def state_features(self, state: Corso) -> torch.Tensor:
        """Retrieve a tensor representing a game state.

        Return shape is arbitrary, shall be compliant with the inference
        mechanism defined in :meth:`__call__`.
        """

    def __call__(self, batch: torch.Tensor) -> tuple[torch.Tensor,
                                                     torch.Tensor]:
        """Prediction of the priors.

        Must return a batch of non-normalized policies (logits) of
        shape ``(batch_size, height * width)`` and a batch of state
        values of shape ``(batch_size, 1)``.
        """


class AZDenseNetwork(nn.Module, SavableModule):
    """Simple dense network for AZ.

    Outputs the tuple ``(policy, value)``, as part of the parameters
    are shared between the policy network and value network. Output
    shapes as described by :meth:`PriorPredictorProtocol.__call__`.
    """

    def __init__(self, board_size=(DEFAULT_BOARD_SIZE, DEFAULT_BOARD_SIZE),
                 shared_layers_sizes: Iterable[int] = (),
                 policy_hidden_layers_sizes: Iterable[int] = (),
                 value_function_hidden_layers_sizes: Iterable[int] = (),
                 num_players=DEFAULT_PLAYER_NUM,
                 dropout=0.0):
        super().__init__()

        self.shared_layers_sizes = tuple(shared_layers_sizes)
        self.policy_hidden_layers_sizes = tuple(policy_hidden_layers_sizes)
        self.value_function_hidden_layers_sizes = tuple(
            value_function_hidden_layers_sizes)
        self.num_players = num_players
        self.dropout = dropout

        board_w, board_h = board_size
        self.board_size = board_size

        # Shared layers
        # W * H * players * 2 + 1 is the input size
        # If the number of shared layers is 0, fall back to this
        # dimentions.
        # Initialize shared weights
        shared_output_size = board_w * board_h * num_players * 2 + 1

        shared_layers = []

        for shared_layer_size in shared_layers_sizes:
            shared_layers.append(nn.Linear(shared_output_size,
                                           shared_layer_size))
            shared_output_size = shared_layer_size

        self.shared_layers = nn.ModuleList(shared_layers)

        # Policy head
        policy_hidden_output_size = shared_output_size

        policy_hidden_layers = []

        for policy_layer_size in policy_hidden_layers_sizes:
            policy_hidden_layers.append(nn.Linear(policy_hidden_output_size,
                                                  policy_layer_size))
            policy_hidden_output_size = policy_layer_size

        self.policy_hidden_layers = nn.ModuleList(policy_hidden_layers)

        # Output of policy head is always width * height
        self.policy_output = nn.Linear(policy_hidden_output_size,
                                       board_w * board_h)

        # Value head
        value_function_hidden_output_size = shared_output_size

        value_function_hidden_layers = []

        for value_function_layer_size in value_function_hidden_layers_sizes:
            value_function_hidden_layers.append(
                nn.Linear(value_function_hidden_output_size,
                          value_function_layer_size))
            value_function_hidden_output_size = value_function_layer_size

        self.value_function_hidden_layers = nn.ModuleList(
            value_function_hidden_layers)

        # Output of policy head is always 1
        self.value_function_output = nn.Linear(
            value_function_hidden_output_size, 1)

    def get_config(self) -> dict:
        """Return a configuration dict: used to save/load the model."""
        return {'board_size': self.board_size,
                'shared_layers': self.shared_layers_sizes,
                'policy_hidden_layers': self.policy_hidden_layers_sizes,
                'value_function_hidden_layers':
                self.value_function_hidden_layers_sizes,
                'num_players': self.num_players}

    def state_features(self, state: Corso) -> torch.Tensor:
        """Retrieve a tensor representing a game state.

        Return shape: ``[1, height * width * num_players * 2 + 1]``.
        """
        # This comes with some necessary reallocations before feeding the
        # structure to the tensor constructor. Things are cached where
        # possible. Time cost of this transformation (5x5 board): ~1.5e-5
        # Organizing the Corso state in a way that is more friendly w.r.t.
        # this representation (e.g. with one hot encoded tuples as cell
        # states instead of the abstract CellState class) is the most viable
        # option after this one.

        board_tensor = torch.Tensor(
            tuple(tuple(map(bitmap_cell, row)) for row in state.board))

        # Concatenate current player information
        board_tensor = torch.cat([board_tensor.flatten(),
                                  torch.tensor([state.player_index - 1])])

        return board_tensor.unsqueeze(0)

    def forward(self, batch: torch.Tensor):
        # Invalid move masking:
        # Reshape input as a grid and collapse it into a bitmap of
        # invalid locations
        with torch.no_grad():
            board_w, board_h = self.board_size

            current_player = batch[:, -1]
            reshaped_input = batch[:, :-1].view(-1, board_h, board_w,
                                                self.num_players * 2)
            # Moves that are invalid because dyed
            invalid_moves_dyed = reshaped_input[:, :, :, [0, 2]].sum(dim=-1)
            # Moves that are invalid because occupied by opponent marble
            # (only works for two players)
            invalid_moves_marble = reshaped_input[
                :, :, :, 1 + 2 * (1 - current_player.int())].flatten(2)

            invalid_moves = (invalid_moves_dyed + invalid_moves_marble).bool()
            invalid_moves = invalid_moves.view(-1, board_w * board_h)

        # Shared layers
        for layer in self.shared_layers:
            batch = F.relu(layer(batch))
            batch = F.dropout(batch, self.dropout, self.training)

        policy_batch = batch
        for layer in self.policy_hidden_layers:
            policy_batch = F.relu(layer(policy_batch))
            policy_batch = F.dropout(policy_batch, self.dropout, self.training)
        policy_batch = self.policy_output(policy_batch)

        # Invalid moves are set to an extremely negative value.
        # Extremely negative logits will results in 0 probability of
        # choosing the action
        policy_batch[invalid_moves] = -1e10

        value_batch = batch
        for layer in self.value_function_hidden_layers:
            value_batch = F.relu(layer(value_batch))
            value_batch = F.dropout(value_batch, self.dropout, self.training)
        value_batch = self.value_function_output(value_batch)

        return policy_batch, value_batch


def puct_siblings(predictions: np.ndarray, counts: np.ndarray,
                  c_puct: float = 1, epsilon: float = 1e-6):
    """Compute PUCT algorithm for siblings.

    As defined by Silver et al. 2017. in "Mastering the game of Go
    without human knowledge".

    The algorithm is computed on a "batch" of siblings. That is,
    ``counts`` shall be the visit counts of sibling nodes in a
    state-action tree.
    """
    return c_puct * predictions * (np.sqrt(counts.sum() + epsilon)
                                   / (1 + counts))


class MCTSNode:
    """"""

    def __init__(self, network: PriorPredictorProtocol, state: Corso):
        self.network: PriorPredictorProtocol = network

        self.state = Corso()

        # All saved metrics are referred to the children of the node
        self.children: list['MCTSNode'] = []
        self.visits = np.array([])
        self.q_values = np.array([])
        self.cumulative_values = np.array([])
        self.actions: list[Action] = []
        self.priors = np.array([])
