import enum
from collections import namedtuple
from dataclasses import dataclass
from typing import Callable, Optional, Union
from itertools import cycle

from corso.model import Player, Corso


GamesResults = tuple[int, int, int]
"""In order: number of draws, wins of player1, wins of player2."""

AgentGamesResults = namedtuple(
    'AgentGamesResults', ['draws_as_p1', 'draws_as_p2', 'wins_as_p1',
                          'wins_as_p2'])
"""Similar to GamesResults but from the agent's point of view.

In order: draws as player1, draws as player2, wins as player1,
wins as player2.
"""


class EvaluationPlayAs(enum.Enum):
    """Possible values for :attr:`EvaluationStrategy.play_as`."""
    PLAYER_ONE = 'one'
    PLAYER_TWO = 'two'
    BOTH = 'both'


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


@dataclass
class AgentEvaluationStrategy:
    """Describe an evaluation strategy to be used during training.

    If ``play_as`` is set to both (default) the same number of games
    will be played as player one as well as player two, hence twice
    the total number of games is played.
    """
    opponent: Callable[[], Player]
    games: int
    play_as: Union[EvaluationPlayAs, str] = EvaluationPlayAs.BOTH
    name: Optional[str] = None
    normalize: bool = True

    def __post_init__(self):
        """Post process enum."""
        if isinstance(self.play_as, EvaluationPlayAs):
            self.play_as = EvaluationPlayAs(self.play_as)

    def evaluate(self, agent: Player,
                 starting_state=Corso()) -> AgentGamesResults:
        """Run evaluation as described by self.

        Return a tuple describing the wins of the agent with respect
        to the opponent. See :attr:`AgentGamesResults` for info.
        """
        opponent = self.opponent()

        player_one_results = (0, 0, 0)
        if (self.play_as == EvaluationPlayAs.PLAYER_ONE
                or self.play_as == EvaluationPlayAs.BOTH):
            player_one_results = evaluate(agent, opponent,
                                          starting_state=starting_state,
                                          n_games=self.games)

        player_two_results = (0, 0, 0)
        if (self.play_as == EvaluationPlayAs.PLAYER_TWO
                or self.play_as == EvaluationPlayAs.BOTH):
            player_two_results = evaluate(opponent, agent,
                                          starting_state=starting_state,
                                          n_games=self.games)

        # If specified, normalize by the total number of games
        divisor = 1
        if self.normalize:
            divisor = self.games

        return AgentGamesResults(player_one_results[0] / divisor,
                                 player_two_results[0] / divisor,
                                 player_one_results[1] / divisor,
                                 player_two_results[2] / divisor)

    def get_name(self) -> str:
        """Return a string name for the experiment.

        When :attr:`name` is ``None``, the string name of
        :attr:`opponent` class is returned instead.
        """
        if self.name is None:
            return self.opponent.__class__.__name__
        return self.name
