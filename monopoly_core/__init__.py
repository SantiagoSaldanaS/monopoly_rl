"""
Monopoly Core Engine - Official Tournament Rules Specification
"""

from monopoly_core.constants import (
    BOARD_SPECS,
    COLOR_GROUP_TILES,
    ALL_PROPERTY_INDICES,
    TileType,
    ColorGroup,
    MAX_HOUSES,
    MAX_HOTELS,
    STARTING_CASH,
    GO_SALARY,
    JAIL_FINE,
    INCOME_TAX,
    LUXURY_TAX,
)
from monopoly_core.board import Board, Tile
from monopoly_core.player import Player
from monopoly_core.auction import conduct_auction
from monopoly_core.game import MonopolyGame

__all__ = [
    "MonopolyGame",
    "Board",
    "Tile",
    "Player",
    "conduct_auction",
    "TileType",
    "ColorGroup",
    "MAX_HOUSES",
    "MAX_HOTELS",
    "STARTING_CASH",
]
