import abc
import enum
import random
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

# Some more common boards
EMPTY_BOARD2X2 = ((EMPTY_CELL, EMPTY_CELL), (EMPTY_CELL, EMPTY_CELL))
EMPTY_BOARD3X3 = ((EMPTY_CELL, EMPTY_CELL, EMPTY_CELL),
                  (EMPTY_CELL, EMPTY_CELL, EMPTY_CELL),
                  (EMPTY_CELL, EMPTY_CELL, EMPTY_CELL))


class Terminal(enum.IntEnum):
    """Terminal states for a game."""
    NOT_TERMINAL = 0
    DRAW = 1
    WON = 2


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
    """Return a mutable version of the given board.

    Each row is still immutable (they are not reallocated), but can now
    be replaced if necessary.
    """
    return list(board)


def _mutable_row(board: MutableBoard, row_index: int) -> MutableBoard:
    """Given a mutable board, substitute a row with a mutable version.

    This causes the reallocation (only) of the selected row.
    """
    board[row_index] = list(board[row_index])
    return board[row_index]


def _consolidate_board(board: MutableBoard, rows: Iterable[int]) -> Board:
    """Given a mutable board, return a consolidated immutable one.

    This causes the reallocation of the selected rows. When
    consolidating, select rows that were previously altered to be
    mutable.
    """
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
                    or cell.player_index == self.player_index
                        and cell.marble):
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

    def _expand(self, row_index: int, column_index: int) -> Board:
        """Expand a marble by reallocating the minimal amount of state.

        Return the new board.
        """
        mutable_board = _mutable_board(self.board)
        cell_state = _cell_state(self.player_index, False)

        mutable_row_indeces = set()
        fringe = [(row_index, column_index)]
        while fringe:
            popped_row, popped_col = fringe.pop()

            if popped_row not in mutable_row_indeces:
                mutable_row_indeces.add(popped_row)
                _mutable_row(mutable_board, popped_row)

            cur_cell = mutable_board[popped_row][popped_col]
            mutable_board[popped_row][popped_col] = cell_state

            # Keep expanding only if selected cell is a marble
            if not cur_cell.marble:
                continue

            if popped_row > 0:
                fringe.append((popped_row - 1, popped_col))
            if popped_row < self.height - 1:
                fringe.append((popped_row + 1, popped_col))
            if popped_col > 0:
                fringe.append((popped_row, popped_col - 1))
            if popped_col < self.width - 1:
                fringe.append((popped_row, popped_col + 1))

        return _consolidate_board(mutable_board, mutable_row_indeces)

    @property
    def terminal(self) -> tuple[Terminal, int]:
        """Retrieve whether the state is terminal.

        Returns a pair ``(terminal_state, winner)``. If the game is not
        terminal or it is a draw, the second value has no meaning.
        In case of a won game, the second value is the winner player's
        index (indexed from 1).
        """
        scores = [0] * (self.player_num + 1)

        for row in self.board:
            for cell in row:
                scores[cell.player_index] += 1

        if scores[0] > 0:
            return Terminal.NOT_TERMINAL, 0

        # Allocating a set is not necessary, but still faster than
        # a pure python loop
        if len(set(scores[1:])) == 1:
            return Terminal.DRAW, 0

        # Argmax: 3+ times faster than a single for loop
        return Terminal.WON, scores.index(max(scores[1:]))

    def __eq__(self, other: 'Corso') -> bool:
        """Check equality based on board and player info."""
        return (self.board == other.board
                and self.player_num == other.player_num
                and self.player_index == other.player_index)


class Player(abc.ABC):
    """Base class for an automated player."""

    @abc.abstractmethod
    def select_action(self, state: Corso) -> Action:
        """Inherit to provide an action selection strategy."""


class RandomPlayer(Player):
    """Select moves randomly."""

    def __init__(self, rng=random.Random()):
        self.rng = rng

    def select_action(self, state: Corso) -> Action:
        """Return a random action."""
        return self.rng.choice(state.actions)
