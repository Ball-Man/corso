"""AlphaZero-like approaches to solve the game."""
import numpy as np

from corso.model import Corso, Action


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

    def __init__(self, network, state: Corso):
        self.network = network

        self.state = Corso()

        # All saved metrics are referred to the children of the node
        self.children: list['MCTSNode'] = []
        self.visits = np.array([])
        self.q_values = np.array([])
        self.cumulative_values = np.array([])
        self.actions: list[Action] = []
        self.priors = np.array([])
