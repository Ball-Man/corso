from corso.model import Corso, Board

MARBLES = ['O', 'A', 'B']
CELLS = ['O', 'a', 'b']


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


def cli_game():
    state = Corso()

    while not state.terminal[0]:
        print_board(state.board)
        input()


if __name__ == '__main__':
    cli_game()
