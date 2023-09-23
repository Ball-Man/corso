"""AlphaZero-like approaches to solve the game.

Current code is designed for two player games.
"""
import random
from typing import Iterable, Protocol, Optional
from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from corso.model import (Corso, Action, DEFAULT_BOARD_SIZE, DEFAULT_PLAYER_NUM,
                         Player)
from corso.utils import SavableModule, bitmap_cell


MCTSTrajectory = list[tuple['MCTSNode', Optional[int]]]


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
                np.arange(0, len(reshaped_input)), :, :,
                1 + 2 * (1 - current_player.int())].flatten(2)

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
        value_batch = F.tanh(self.value_function_output(value_batch))

        return policy_batch, value_batch


def puct_siblings(priors: np.ndarray, counts: np.ndarray,
                  c_puct: float = 1, epsilon: float = 1e-6):
    """Compute PUCT algorithm for siblings.

    As defined by Silver et al. 2017. in "Mastering the game of Go
    without human knowledge".

    The algorithm is computed on a "batch" of siblings. That is,
    ``counts`` shall be the visit counts of sibling nodes in a
    state-action tree.
    """
    return c_puct * priors * (np.sqrt(counts.sum() + epsilon) / (1 + counts))


class MCTSNode:
    """MTCS variation as described by AZ.

    Single threaded version.
    """

    def __init__(self, network: PriorPredictorProtocol, state: Corso,
                 parent: Optional['MCTSNode'] = None, value: float = 0,
                 priors: np.ndarray = np.array([])):
        self.network: PriorPredictorProtocol = network

        self.state = state
        self.parent: Optional['MCTSNode'] = parent
        self.value: float = value

        # All saved metrics are referred to the children of the node
        self.children: list['MCTSNode'] = []
        self.visits = np.array([])
        self.q_values = np.array([])
        self.cumulative_values = np.array([])
        self.actions: list[Action] = []
        self.priors = priors

    @classmethod
    def create_root(cls, network: PriorPredictorProtocol,
                    state: Corso) -> 'MCTSNode':
        """Generate a root node, initializing priors from the network."""
        with torch.no_grad():
            priors, value = network(network.state_features(state))

        return cls(network, state, value=value.item(),
                   priors=priors.numpy().squeeze())

    def select(self) -> MCTSTrajectory:
        """Explore existing tree and select node to expand.

        Return the trajectory for the selection (used for backup).
        Return format is
        ``[(root_node, selected_index), ..., (node_to_expand, None)]``.
        """
        selected_node = self
        trajectory = []

        # Compute factors to shift Q values based on current player
        # 1 if current player is player1, -1 if player2.
        initial_player_factor = -2 * self.state.player_index + 3
        player_factor = cycle((initial_player_factor, -initial_player_factor))

        while selected_node.children:
            # Q + U
            bounds = (next(player_factor) * selected_node.q_values
                      + puct_siblings(selected_node.priors,
                                      selected_node.visits))

            selected_index = bounds.argmax()
            selected_node.visits[selected_index] += 1

            # Push current node with the selected child index
            trajectory.append((selected_node, selected_index))

            # Select new node
            selected_node = selected_node.children[selected_index]

        trajectory.append((selected_node, None))
        return trajectory

    def expand(self):
        """Expand children.

        Completely populate children nodes and their metrics. Priors
        are predicted through :attr:`network`. This include the rollout
        step which is replaced with neural network predictions of the
        state value function.
        """
        if self.state.terminal[0]:
            return

        # Expand self actions and select the correct subset of priors
        self.actions = self.state.actions
        action_indeces = [action.row * self.state.width + action.column
                          for action in self.actions]
        self.priors = self.priors[action_indeces]

        # Compute priors and state values
        children_states = [self.state.step(action) for action
                           in self.actions]

        children_tensor = torch.cat([self.network.state_features(state) for
                                     state in children_states])
        with torch.no_grad():
            logits, predicted_values = self.network(children_tensor)

            all_priors = torch.softmax(logits, dim=-1).numpy()

        # If the generated node is terminal, use the true game outcome
        # as backup value (only works for two players)
        terminal_map = np.array(
            [state.terminal for state in children_states], dtype=int)
        values = predicted_values.numpy().flatten()
        is_terminal = terminal_map[:, 0].astype(np.bool_)
        # Map winner to {-1, 1}
        values[is_terminal] = -2 * terminal_map[is_terminal, 1] + 3

        self.children = [MCTSNode(self.network, state, self, value, priors)
                         for state, value, priors
                         in zip(children_states, values, all_priors)]

        # Initialize other metrics
        self.visits = np.ones_like(self.children, dtype=int)
        self.q_values = np.zeros_like(self.children)
        self.cumulative_values = np.zeros_like(self.children)

    @staticmethod
    def backup(trajectory: MCTSTrajectory):
        """Backup value along the given trajectory."""
        backup_value = trajectory[-1][0].value

        for node, child_index in trajectory:
            if child_index is None:
                break

            # Move here increment to visits?
            node.cumulative_values[child_index] += backup_value
            node.q_values[child_index] = (node.cumulative_values[child_index]
                                          / node.visits[child_index])

    def search(self):
        """Run selection, expansion, rollout and backup.

        Expansion and rollout steps are unified for convenience. Rollout
        is replaced with immediate predictions from the neural network.
        """
        trajectory = self.select()
        trajectory[-1][0].expand()
        self.backup(trajectory)


def visits_policy(mcts_root: MCTSNode,
                  temperature: float = 1.,
                  epsilon: float = 1e-2) -> np.ndarray:
    """Return normalized policy, based on exponential visit counts.

    Temperature controls exploration (temp of zero: no exploration,
    temp of one: exploration is proportional to the original visit
    counts).

    Output is an array of shape ``(n_actions,)`` where ``n_actions`` is
    the number of legal actions according to the given ``mcts_root`` (
    i.e. ``len(mcts_root.actions)``).
    """
    exponential_visits = mcts_root.visits ** (1 / (temperature + epsilon))
    return exponential_visits / exponential_visits.sum()


class AZPlayer(Player):
    """Corso player based on AZ agent.

    Works for two player games only.
    """

    def __init__(self, network: PriorPredictorProtocol,
                 mcts_simulations: int,
                 temperature: float = 1.,
                 seed=None):
        self.network = network
        self.mcts_simulations = mcts_simulations
        self.temperature = temperature

        self.rng = random.Random(seed)

        self._mcts_tree: Optional[MCTSNode] = None

        self.last_policy = np.array([])

    def select_action(self, state: Corso,
                      mcts_tree: Optional[MCTSNode] = None) -> Action:
        """Select action based on MCTS and internal pretrained network.

        If ``mcts_tree`` is given, use it as starting node for the
        simulations. It must be true that ``mcts_tree.state == state``.
        Ultimately, it will also replace the internal tree,
        which is discarded.
        Otherwise, the internally stored tree is
        used. This assumes a two player game, and that the internal tree
        root is either uninitilized or positioned in a way such that
        ``state`` can be found in its immediate children. This is only
        true if this object is used consistently in a two player game.
        If ``state`` is not found in the immediate children, a new
        internal tree is created. This ensures correctness even in the
        case of inconsistent queries, but in such circumstances loses
        efficiency and potentially optimality.

        After selection of the action is done, :attr:`last_policy` is
        populated, which can be used to retrieve the policy that was
        used to make such decision. This is mostly useful for training.
        """
        if mcts_tree is None:
            children_states = ()
            if self._mcts_tree is not None:
                children_states = [node.state for node
                                   in self._mcts_tree.children]

            if state in children_states:
                # Be aware of the double list lookup
                # (state in children, children.index)
                mcts_tree = self._mcts_tree.children[
                    children_states.index(state)]
            else:
                mcts_tree = MCTSNode.create_root(self.network, state)

        for _ in range(self.mcts_simulations):
            mcts_tree.search()

        self.last_policy = visits_policy(mcts_tree)

        selected_index, = self.rng.choices(
            range(len(mcts_tree.actions)), weights=self.last_policy,
            k=1)

        # Move tree search to the selected child in order to preserve
        # tree during turns
        self._mcts_tree = mcts_tree.children[selected_index]

        return mcts_tree.actions[selected_index]
