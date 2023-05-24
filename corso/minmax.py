"""Simple minmax agent definition for the two players game of Corso."""
from corso.model import Corso, CellState

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


def minmax_score(state: Corso, heuristic=heuristic, depth=3) -> float:
    """Run a minmax search for a maximum given depth and a states score."""
    # Base case, return value computed by the heuristic
    if depth <= 0 or state.terminal[0]:
        return heuristic(state)

    # Selection method is given by the current player
    select = min
    if state.player_index == 1:
        select = max

    # Compute scores of immediate states and retrieve the argmax/argmin
    # TODO: compare to an iterative solution? This recursive one is
    #       very slim and consequently good in terms of algorithmical
    #       constants. However, how do the recusion stack/func call
    #       management constants scale?
    return select(
        map(lambda a: minmax_score(state.step(a), heuristic, depth - 1),
            state._iter_actions()))
