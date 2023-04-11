from functools import lru_cache
from collections import namedtuple
from typing import Iterable, Sequence

DEFAULT_BOARD_SIZE = 5
DEFAULT_PLAYER_NUM = 2
Action = namedtuple('Action', ['player_index', 'row', 'column'])
CellState = namedtuple('CellState', ['player_index', 'marble'])
EMPTY_CELL = CellState(0, 0)
Board = tuple[tuple[CellState, ...], ...]
MutableBoard = Sequence[Sequence[CellState]]
EMPTY_BOARD = tuple([tuple([EMPTY_CELL] * DEFAULT_BOARD_SIZE)]
                    * DEFAULT_BOARD_SIZE)


@lru_cache()
def _cell_state(player_index: int, marble: bool) -> CellState:
    """Retrieve a cached cell state.

    Args:
        player_index: 0 means no player (empty cell).
        marble: True if a full marble is placed. False if its an
            exploded one.
    """
    return CellState(player_index, marble)


@lru_cache()
def _action(player_index: int, row: int, col: int) -> Action:
    """Retrieve a cached action."""
    return Action(player_index, row, col)


def _mutable_board(board: Board) -> MutableBoard:
    """ """
    return list(board)


def _mutable_row(board: MutableBoard, row_index: int) -> MutableBoard:
    """ """
    board[row_index] = list(board[row_index])
    return board[row_index]


def _consolidate_board(board: MutableBoard, rows: Iterable[int]) -> Board:
    """ """
    for row in rows:
        board[row] = tuple(board[row])

    return tuple(board)


class Corso:
    """Corso game model."""

    # def __init__(self, width=DEFAULT_BOARD_SIZE, height=DEFAULT_BOARD_SIZE,
    #              player_num=DEFAULT_PLAYER_NUM):
    #     self.board: Board = EMPTY_BOARD
    #     self.width = width
    #     self.height = height
    #     self.player_num = DEFAULT_PLAYER_NUM
    #     self.player_index = 1

    def __init__(self, board: Board = EMPTY_BOARD,
                 player_num: int = DEFAULT_PLAYER_NUM, player_index: int = 1):
        """Create game s tate from a preinitialized board."""
        self.board = board
        self.width = len(board[0])
        self.height = len(board)
        self.player_num = player_num
        self.player_index = player_index

    def _iter_actions(self) -> Iterable[Action]:
        """Retrieve an iterable yielding all legal actions."""
        for row_index, row in enumerate(self.board):
            for col_index, cell in enumerate(row):
                if (cell.player_index == 0
                        or cell.player_index == self.player_index):
                    yield _action(self.player_index, row_index, col_index)

    @property
    def actions(self) -> list[Action]:
        """Retrieve legal actions."""
        return list(self._iter_actions())

    def step(self, action: Action):
        """Apply action.

        No sanity checks for convenience. Check if moves are in
        :attr:`actions` before applying them.
        """
        _, row, column = action

        if self.board[row][column].marble:
            new_board = self._expand(row, column)
        else:
            new_board = self._set(row, column, True)

        new_player = self.next_player()

        return self.__class__(new_board, self.player_num, new_player)

    def next_player(self):
        """Return next player."""
        return self.player_index % self.player_num + 1

    def _set(self, row: int, column: int, marble: bool) -> Board:
        """Set cell state by reallocating the minimal amount of state.

        Return the new board.
        """
        mutable_board = _mutable_board(self.board)
        new_row = _mutable_row(mutable_board, row)

        new_row[column] = _cell_state(self.player_index, marble)

        return _consolidate_board(mutable_board, (row,))

    def _expand(self, row: int, column: int):
        """ """
