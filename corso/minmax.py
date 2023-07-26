"""Simple minmax agent definition for the two players game of Corso."""
import random
import numpy as np

from corso.model import Corso, CellState, Player, Action

TERMINAL_SCORE = 100000.
P1_MARBLE_STATE = CellState(1, True)
P2_MARBLE_STATE = CellState(2, True)
P1_DYE_STATE = CellState(1, False)
P2_DYE_STATE = CellState(2, False)


def heuristic(state: Corso) -> float:
    """Retrieve the heuristic value of a state.

    Value is signed and represents the absolute advantage of a player
    (positive player 1, negative player 2).
    """
    terminal, winner = state.terminal
    # In case of termination propagate certainty of winning/losing
    if terminal:
        if winner == 1:
            return TERMINAL_SCORE
        return -TERMINAL_SCORE

    p1_marbles = sum(line.count(P1_MARBLE_STATE) for line in state.board)
    p2_marbles = sum(line.count(P2_MARBLE_STATE) for line in state.board)
    p1_dyes = sum(line.count(P1_DYE_STATE) for line in state.board)
    p2_dyes = sum(line.count(P2_DYE_STATE) for line in state.board)

    # Simple heuristic in function of how many cells are occupied by
    # each player
    return p1_marbles + 0.7 * p1_dyes - p2_marbles - 0.7 * p2_dyes


def minmax_score(state: Corso, heuristic=heuristic, depth=3,
                 alpha=-TERMINAL_SCORE, beta=TERMINAL_SCORE,
                 cache=None) -> float:
    """Run a minmax search for a maximum given depth and a states score."""
    # Exploit cache if present
    if cache is None:
        cache = {}

    cache_key = (state.board, state.player_index)
    if cache_key in cache:
        return cache[cache_key]

    # Base case, return value computed by the heuristic
    if depth <= 0 or state.terminal[0]:
        return heuristic(state)

    # Selection method is given by the current player
    select = min
    if state.player_index == 1:
        select = max

    # Compute scores of immediate states and retrieve the argmax/argmin
    score = None
    for action in state._iter_actions():
        action_score = minmax_score(state.step(action), heuristic, depth - 1,
                                    alpha, beta, cache)

        if score is None:
            score = action_score
        score = select(score, action_score)

        if state.player_index == 1:
            alpha = max(alpha, action_score)
        else:
            beta = min(beta, action_score)

        if alpha >= beta:
            break

    # Update cache
    reversed_cache_key = (tuple(reversed(state.board)), state.player_index)
    cache[cache_key] = score
    cache[reversed_cache_key] = score

    return score


class MinMaxPlayer(Player):
    """Player employing a minmax strategy at fixed depth.

    The player uses heuristic scores computed from the minmax as a
    policy, sampling the action based on the scores.

    Temperature handles exploration: high temperatures morph the policy
    towards a uniform distribution, lower temperatures morph the policy
    towards an optimal choice. Due to the very high heuristic scores
    assigned to winning positions (terminal states), unless the
    temperature is extremely high the agent will always make a certainly
    winning move if one is found, sampling uniformly between equally
    winning moves. Similarly, it will never play a certainly losing
    action if found.
    """

    def __init__(self, depth=3, heuristic=heuristic, temperature=1.,
                 verbose=False, rng=random.Random()):
        self.depth = depth - 1
        self.heuristic = heuristic
        self.temperature = temperature
        self.verbose = verbose
        self.rng = rng

    def select_action(self, state: Corso) -> Action:
        """Run a minmax search and return best scoring action."""
        actions = state.actions
        scores = np.fromiter(
            map(lambda s: minmax_score(s, self.heuristic,
                                       self.depth),
                (state.step(a) for a in actions)),
            dtype=np.float32)

        # In case of a min player, maximize negative of the score
        if state.player_index == 2:
            scores = -scores

        # Compute policy from minmax scores and sample
        policy = softmax(scores / self.temperature)

        if self.verbose:
            print(tuple(zip(policy, actions)))

        return self.rng.choices(actions, policy)[0]


def softmax(x: np.ndarray) -> np.ndarray:
    """Stable softmax, decoupled from torch."""
    shifted_x = x - x.max()
    numerator = np.exp(shifted_x)
    return numerator / numerator.sum()
