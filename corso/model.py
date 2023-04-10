from functools import lru_cache
from collections import namedtuple
from typing import Iterable

DEFAULT_BOARD_SIZE = 5
DEFAULT_PLAYER_NUM = 2
Action = namedtuple('Action', ['player_index', 'row', 'column'])
CellState = namedtuple('CellState', ['player_index', 'marble'])
EMPTY_CELL = CellState(0, 0)


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


class Corso:
    """Corso game model."""

    def __init__(self, width=DEFAULT_BOARD_SIZE, height=DEFAULT_BOARD_SIZE,
                 player_num=DEFAULT_PLAYER_NUM):
        self.board: tuple[tuple[CellState, ...], ...] = tuple(
            [tuple([EMPTY_CELL] * width)] * height)
        self.width = width
        self.height = height
        self.player_num = DEFAULT_PLAYER_NUM
        self.player_index = 1

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
