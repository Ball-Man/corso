"""Play a CLI game of Corso."""
import argparse
from itertools import cycle

from corso.model import Corso, Board, Action, Player, EMPTY_CELL

MARBLES = ['O', 'A', 'B']
CELLS = ['O', 'a', 'b']

# CLI
DESCRIPTION = __doc__
WIDTH_DESCRIPTION = 'Width of the game grid, defaults to 5.'
HEIGHT_DESCRIPTION = 'Height of the game grid, defaults to 5.'


class Namespace:
    """Custom namespace for CLI arguments."""
    width: int = 5
    height: int = 5


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


def get_input_pair(state: Corso) -> tuple[int, int]:
    """Retrieve row, column input from user."""
    while True:
        input_string = input()
        values = input_string.split()

        if len(values) != 2:
            print('Insert two values separated by a whitespace')
            continue

        integers_ok = True
        try:
            row, col = map(int, values)
            integers_ok &= 0 <= row < state.height
            integers_ok &= 0 <= col < state.width
        except ValueError:
            integers_ok = False

        if not integers_ok:
            print('Input values must be integers.\n'
                  f'The first value must be in the range [0, {state.height}). '
                  f'The second value must be in the range [0, {state.width}).')
            continue

        return row, col


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


def cli_game(player1: Player = CLIPlayer(), player2: Player = CLIPlayer(),
             starting_state: Corso = Corso()):
    state = starting_state

    players = cycle((player1, player2))
    for _ in range(starting_state.player_index - 1):
        next(players)

    while not state.terminal[0]:
        player = next(players)

        print('Player', state.player_index)
        print_board(state.board)
        state = state.step(player.select_action(state))

    print_board(state.board)
    print(state.terminal)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)

    parser.add_argument('-w', '--width', required=False, type=int)
    parser.add_argument('-H', '--height', required=False, type=int)

    namespace = parser.parse_args(namespace=Namespace())

    if namespace.width <= 0 or namespace.height <= 0:
        raise ValueError('Size of the board must be positive.')

    game_board = tuple([tuple([EMPTY_CELL] * namespace.width)]
                       * namespace.height)

    cli_game(starting_state=Corso(game_board))
