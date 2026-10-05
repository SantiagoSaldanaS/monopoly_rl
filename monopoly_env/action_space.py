"""
Action Space and Action Masking for Monopoly Reinforcement Learning.
Enforces strict legal action validity according to official Hasbro rules.
Features group-level development to eliminate asymmetric building deadlocks.
"""

from enum import IntEnum
import numpy as np
from typing import TYPE_CHECKING, Optional, Set
from monopoly_core.constants import (
    ALL_PROPERTY_INDICES,
    COLOR_GROUP_TILES,
    ColorGroup,
    JAIL_FINE,
)

if TYPE_CHECKING:
    from monopoly_core.game import MonopolyGame

NUM_PROPERTIES = len(ALL_PROPERTY_INDICES)  # 28

BUILDABLE_GROUPS = [
    ColorGroup.BROWN,       # 0
    ColorGroup.SKY_BLUE,    # 1
    ColorGroup.PINK,        # 2
    ColorGroup.ORANGE,      # 3
    ColorGroup.RED,         # 4
    ColorGroup.YELLOW,      # 5
    ColorGroup.GREEN,       # 6
    ColorGroup.DARK_BLUE,   # 7
]
NUM_GROUPS = len(BUILDABLE_GROUPS)  # 8


class Action(IntEnum):
    END_OR_ROLL = 0
    BUY_PROPERTY = 1
    DECLINE_TO_AUCTION = 2
    PAY_JAIL_FINE = 3
    USE_JAIL_CARD = 4
    # 5 .. 12: Build houses on color group 0..7 (Brown .. Dark Blue)
    # 13 .. 20: Sell building on color group 0..7
    # 21 .. 48: Mortgage property 0..27
    # 49 .. 76: Unmortgage property 0..27


BUILD_GROUP_START = 5
SELL_GROUP_START = BUILD_GROUP_START + NUM_GROUPS         # 13
MORTGAGE_START = SELL_GROUP_START + NUM_GROUPS           # 21
UNMORTGAGE_START = MORTGAGE_START + NUM_PROPERTIES       # 49
TOTAL_ACTIONS = UNMORTGAGE_START + NUM_PROPERTIES        # 77 actions

# Backwards compatibility aliases
BUILD_START = BUILD_GROUP_START
SELL_START = SELL_GROUP_START


def get_action_mask(
    game: "MonopolyGame",
    player_id: int,
    pending_decision: str = "MANAGEMENT",
    modified_props: Optional[Set[int]] = None,
) -> np.ndarray:
    """
    Computes a binary action mask where 1 = Legal action, 0 = Illegal.
    
    Pending decisions:
    - "BUY_OR_AUCTION": Must either buy or pass to auction.
    - "JAIL": In jail, choose pay fine, use card, or roll.
    - "MANAGEMENT": Normal turn phase (group-build, group-sell, mortgage, unmortgage, or roll/end).
    """
    mask = np.zeros(TOTAL_ACTIONS, dtype=np.int8)
    player = game.players[player_id]
    board = game.board

    if player.is_bankrupt:
        mask[Action.END_OR_ROLL] = 1
        return mask

    if pending_decision == "BUY_OR_AUCTION":
        tile = board.tiles[player.position]
        if player.cash >= tile.price:
            mask[Action.BUY_PROPERTY] = 1
        mask[Action.DECLINE_TO_AUCTION] = 1
        return mask

    if pending_decision == "JAIL":
        mask[Action.END_OR_ROLL] = 1  # Try rolling doubles
        if player.cash >= JAIL_FINE:
            mask[Action.PAY_JAIL_FINE] = 1
        if player.get_out_of_jail_cards > 0:
            mask[Action.USE_JAIL_CARD] = 1
        return mask

    # Default phase: PROPERTY_MANAGEMENT
    mask[Action.END_OR_ROLL] = 1  # Always allowed to roll or end turn

    # Jail actions if currently in jail
    if player.in_jail:
        if player.cash >= JAIL_FINE:
            mask[Action.PAY_JAIL_FINE] = 1
        if player.get_out_of_jail_cards > 0:
            mask[Action.USE_JAIL_CARD] = 1

    # 1. Group-level building & selling for the 8 buildable color groups (Hasbro rules)
    for g_idx, group in enumerate(BUILDABLE_GROUPS):
        indices = COLOR_GROUP_TILES[group]

        # Building on group: Hasbro rules
        if board.owns_full_group(player_id, group):
            # Rule: Cannot build if any property in the group is mortgaged
            if not any(board.tiles[i].is_mortgaged for i in indices):
                house_cost = board.tiles[indices[0]].house_cost
                min_dev = min(board.tiles[i].num_houses + (5 if board.tiles[i].num_hotels else 0) for i in indices)
                if min_dev < 5 and player.cash >= house_cost:
                    if min_dev == 4:
                        if board.available_hotels > 0:
                            mask[BUILD_GROUP_START + g_idx] = 1
                    else:
                        if board.available_houses > 0:
                            mask[BUILD_GROUP_START + g_idx] = 1

        # Selling building on group: Hasbro rules
        max_dev = max(board.tiles[i].num_houses + (5 if board.tiles[i].num_hotels else 0) for i in indices)
        if max_dev > 0 and board.owns_full_group(player_id, group):
            mask[SELL_GROUP_START + g_idx] = 1

    # 2. Mortgaging / Unmortgaging for each of the 28 properties (Hasbro rules)
    for i, tile_idx in enumerate(ALL_PROPERTY_INDICES):
        if modified_props and tile_idx in modified_props:
            continue

        tile = board.tiles[tile_idx]

        # Pure Hasbro rules: unimproved properties owned by player can be mortgaged
        if board.can_mortgage(tile_idx, player_id):
            mask[MORTGAGE_START + i] = 1

        # Pure Hasbro rules: mortgaged properties owned by player can be unmortgaged if cash >= cost
        unmortgage_cost = int(tile.mortgage_value * 1.1)
        if board.can_unmortgage(tile_idx, player_id) and player.cash >= unmortgage_cost:
            mask[UNMORTGAGE_START + i] = 1

    return mask

