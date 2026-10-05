"""
Player state representation and asset liquidity management.
Audited against Hasbro official rules and DARPA GNOME schema.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from monopoly_core.constants import (
    STARTING_CASH,
    JAIL_FINE,
    UNMORTGAGE_FEE_RATE,
    TileType,
    ColorGroup,
)

if TYPE_CHECKING:
    from monopoly_core.board import Board


@dataclass
class Player:
    player_id: int
    name: str
    cash: int = STARTING_CASH
    position: int = 0
    in_jail: bool = False
    jail_turns: int = 0
    get_out_of_jail_cards: int = 0
    consecutive_doubles: int = 0
    is_bankrupt: bool = False
    agent: object = None

    def net_worth(self, board: "Board") -> int:
        """
        Total asset valuation:
        Cash + Face value of properties + Cost of all buildings.
        """
        if self.is_bankrupt:
            return 0
        total = self.cash
        for tile in board.tiles:
            if tile.owner == self.player_id:
                if not tile.is_mortgaged:
                    total += tile.price
                else:
                    total += tile.mortgage_value
                total += tile.total_buildings * tile.house_cost
        return total

    def max_raisable_cash(self, board: "Board") -> int:
        """
        Calculates the theoretical maximum cash a player can liquidate immediately:
        Current cash + 50% building resale value + mortgage value of unmortgaged properties.
        """
        if self.is_bankrupt:
            return 0
        total = self.cash
        for tile in board.tiles:
            if tile.owner == self.player_id:
                # Buildings sell back to the bank for 50% of purchase price
                total += tile.total_buildings * (tile.house_cost // 2)
                if not tile.is_mortgaged:
                    total += tile.mortgage_value
        return total

    def liquidate_assets_to_cover(self, required_amount: int, board: "Board") -> bool:
        """
        Automated asset liquidation to avoid bankruptcy.
        Steps:
        1. Sells houses/hotels evenly across monopolies at 50% value.
        2. Mortgages unmortgaged properties.
        Returns True if the required amount was successfully raised, False if bankrupt.
        """
        if self.cash >= required_amount:
            return True

        # Phase 1: Sell houses/hotels across all properties
        while self.cash < required_amount:
            # Find eligible building to sell (must respect even breakdown)
            sold_any = False
            for tile in sorted(
                [t for t in board.tiles if t.owner == self.player_id and t.total_buildings > 0],
                key=lambda t: t.house_cost,
                reverse=True,
            ):
                if board.can_sell_building(tile.index, self.player_id):
                    # Sell 1 building
                    sell_value = tile.house_cost // 2
                    if tile.num_hotels == 1:
                        if board.available_houses >= 4:
                            tile.num_hotels = 0
                            tile.num_houses = 4
                            board.available_hotels += 1
                            board.available_houses -= 4
                            self.cash += sell_value
                        else:
                            # If bank doesn't have 4 houses, entire hotel + 4 houses liquidated
                            tile.num_hotels = 0
                            tile.num_houses = 0
                            board.available_hotels += 1
                            self.cash += sell_value * 5
                    else:
                        tile.num_houses -= 1
                        board.available_houses += 1
                        self.cash += sell_value

                    sold_any = True
                    break

            if not sold_any:
                break

        if self.cash >= required_amount:
            return True

        # Phase 2: Mortgage properties
        for tile in sorted(
            [t for t in board.tiles if t.owner == self.player_id and not t.is_mortgaged],
            key=lambda t: t.mortgage_value,
            reverse=True,
        ):
            if board.can_mortgage(tile.index, self.player_id):
                tile.is_mortgaged = True
                self.cash += tile.mortgage_value
                if self.cash >= required_amount:
                    return True

        return self.cash >= required_amount
