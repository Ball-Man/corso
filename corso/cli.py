"""Play a CLI game of Corso."""
import argparse
import re
from dataclasses import dataclass, field
from itertools import cycle, chain
from string import ascii_lowercase, ascii_uppercase

from corso.model import (Corso, Board, Action, Player, RandomPlayer,
                         EMPTY_CELL, Terminal)
from corso.minmax import MinMaxPlayer


MARBLES = ('O',) + tuple(ascii_uppercase)
CELLS = ('O',) + tuple(ascii_lowercase)

# CLI
DESCRIPTION = __doc__
WIDTH_DESCRIPTION = 'Width of the game grid, defaults to 5.'
HEIGHT_DESCRIPTION = 'Height of the game grid, defaults to 5.'
PLAYER_DESCRIPTION = """\
Specify one or more player types for the game. Accepted player types
are: "user", "random", "mmX". "user" is desigend for human input from
CLI. "random" plays completely randomly. "mmX" is a MinMax player, where
X specifies the depth of the search. X must be an integer greater than
1. If omitted, defaults to 3. A suitable range goes from 1 to 6.
Generally, a deeper search leads to a stronger player. This option
can be specified multiple times to define the player order. A minumum
of two players is required. If omitted, "user" type players are
automatically added.
"""

MINMAX_PLAYER_RE = re.compile(r'mm((?:[1-9]\d*)|)')
MINMAX_DEFAULT_DEPTH = 3
MINMAX_DEFAULT_TEMPERATURE = 1e-5

USER_PROMPT = '> '


def print_board(board: Board):
    """Print board on CLI."""
    output_board = [([''] * len(board[0])) for _ in range(len(board))]

    for row_index, row in enumerate(board):
        for cell_index, cell in enumerate(row):
            symbols = CELLS
            if cell.marble:
                symbols = MARBLES

            output_board[row_index][cell_index] = symbols[cell.player_index]

    print('\n'.join(map(' '.join, output_board)))


def format_action(action: Action) -> str:
    """Return a string representing the action for CLI visualization."""
    return f'{action.row + 1} {action.column + 1}'


def format_terminal(terminal_pair: tuple[Terminal, int]) -> str:
    """Return a string representing the exodus of the match for CLI."""
    terminal, winner = terminal_pair

    if not terminal:
        return 'The game is not over'
    elif terminal == Terminal.DRAW:
        return "It's a draw"

    return f'Player {winner} wins'


def get_input_pair(state: Corso) -> tuple[int, int]:
    """Retrieve row, column input from user."""
    while True:
        input_string = input(USER_PROMPT)
        values = input_string.split()

        if len(values) != 2:
            print('Insert two values separated by a whitespace '
                  '(row and column)')
            continue

        integers_ok = True
        try:
            row, col = map(int, values)
            integers_ok &= 1 <= row <= state.height
            integers_ok &= 1 <= col <= state.width
        except ValueError:
            integers_ok = False

        if not integers_ok:
            print('Input values must be integers.\n'
                  f'The first value must be in the range [1, {state.height}]. '
                  f'The second value must be in the range [1, {state.width}].')
            continue

        return row - 1, col - 1


def get_action(state: Corso) -> Action:
    """Retrieve an action from input.

    Refuses actions that are not legal.
    """
    while True:
        row, col = get_input_pair(state)
        candidate_action = Action(state.player_index, row, col)

        # Accept legal action
        if candidate_action in state.actions:
            return candidate_action

        print('Invalid move')


class CLIPlayer(Player):
    """CLI player that reads stdin."""

    def select_action(self, state: Corso) -> Action:
        """Select action based on stdin."""
        return get_action(state)


@dataclass
class Namespace:
    """Custom namespace for CLI arguments."""
    width: int = 5
    height: int = 5
    players: list[Player] = field(default_factory=list)


def cli_game(player1: Player = CLIPlayer(), player2: Player = CLIPlayer(),
             *other_players: Player,
             starting_state: Corso = Corso()):
    """Start a CLI game."""

    provided_players = 2 + len(other_players)
    assert starting_state.player_num == provided_players, (
        'The given starting state is configured for '
        f'{starting_state.player_num} players, but {provided_players} were '
        'provided')

    state = starting_state

    players = cycle(chain((player1, player2), other_players))
    for _ in range(starting_state.player_index - 1):
        next(players)

    while not state.terminal[0]:
        player = next(players)

        print_board(state.board)
        print()

        action = player.select_action(state)
        print(f'Player {state.player_index}: {format_action(action)}')
        state = state.step(action)

    print_board(state.board)
    print(format_terminal(state.terminal))


def parse_player(player_type: str) -> Player:
    """Create a player based on its string description.

    Supported entries::
    * ``random``, a random player.
    * ``user``, user player from CLI.
    * ``mmX``, a MinMax player, with depth X (must be either a positive
        number or omitted). ``mm`` defaults to a depth of 3. MinMax
        players are only suitable for two-player games.

    The parsing process is case insensitive.
    """
    player_type = player_type.lower()

    if player_type == 'random':
        return RandomPlayer()
    elif player_type == 'user':
        return CLIPlayer()
    elif (mm_match := MINMAX_PLAYER_RE.fullmatch(player_type)):
        try:
            depth = int(mm_match.group(1))
        except ValueError:
            depth = MINMAX_DEFAULT_DEPTH

        # What about a more complex syntax to account for temperature and
        # seeding?
        return MinMaxPlayer(depth, temperature=MINMAX_DEFAULT_TEMPERATURE)

    raise ValueError(f'Player type "{player_type}" is not supported. '
                     'Supported types are: "random", "user".')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)

    parser.add_argument('-w', '--width', required=False, type=int,
                        help=WIDTH_DESCRIPTION)
    parser.add_argument('-H', '--height', required=False, type=int,
                        help=HEIGHT_DESCRIPTION)
    parser.add_argument('-p', '--player', required=False, type=parse_player,
                        help=PLAYER_DESCRIPTION,
                        nargs='*', action='extend', metavar='PLAYER_TYPE',
                        dest='players')

    namespace = parser.parse_args(namespace=Namespace())

    if namespace.width <= 0 or namespace.height <= 0:
        raise ValueError('Size of the board must be positive.')

    game_board = tuple([tuple([EMPTY_CELL] * namespace.width)]
                       * namespace.height)

    cli_game(*namespace.players,
             starting_state=Corso(game_board, max(2, len(namespace.players))))
