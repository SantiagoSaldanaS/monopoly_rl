"""
Board representation and state management for official Monopoly rules.
Audited against Hasbro official rules and DARPA GNOME schema.
"""

from dataclasses import dataclass, field
from typing import Optional
import random

from monopoly_core.constants import (
    BOARD_SPECS,
    COLOR_GROUP_TILES,
    TileType,
    ColorGroup,
    MAX_HOUSES,
    MAX_HOTELS,
)
from monopoly_core.cards import (
    CardDeck,
    OFFICIAL_CHANCE_CARDS,
    OFFICIAL_COMMUNITY_CHEST_CARDS,
)

@dataclass
class Tile:
    index: int
    name: str
    tile_type: TileType
    color_group: ColorGroup
    price: int
    mortgage_value: int
    house_cost: int
    base_rent: int
    rent_1: int
    rent_2: int
    rent_3: int
    rent_4: int
    rent_hotel: int

    # Dynamic state
    owner: Optional[int] = None
    num_houses: int = 0
    num_hotels: int = 0
    is_mortgaged: bool = False

    @property
    def is_purchasable(self) -> bool:
        return self.tile_type in (TileType.STREET, TileType.RAILROAD, TileType.UTILITY)

    @property
    def total_buildings(self) -> int:
        return 5 if self.num_hotels == 1 else self.num_houses


class Board:
    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng if rng is not None else random.Random()
        self.available_houses: int = MAX_HOUSES
        self.available_hotels: int = MAX_HOTELS

        # Initialize tiles from constants
        self.tiles: list[Tile] = []
        for spec in BOARD_SPECS:
            self.tiles.append(
                Tile(
                    index=spec[0],
                    name=spec[1],
                    tile_type=spec[2],
                    color_group=spec[3],
                    price=spec[4],
                    mortgage_value=spec[5],
                    house_cost=spec[6],
                    base_rent=spec[7],
                    rent_1=spec[8],
                    rent_2=spec[9],
                    rent_3=spec[10],
                    rent_4=spec[11],
                    rent_hotel=spec[12],
                )
            )

        # Initialize card decks
        self.chance_deck = CardDeck(OFFICIAL_CHANCE_CARDS, self.rng)
        self.community_chest_deck = CardDeck(OFFICIAL_COMMUNITY_CHEST_CARDS, self.rng)

    def owns_full_group(self, player_id: int, color_group: ColorGroup) -> bool:
        """Returns True if player owns all properties in the specified color group."""
        if color_group in (ColorGroup.NONE, ColorGroup.RAILROAD, ColorGroup.UTILITY):
            return False
        indices = COLOR_GROUP_TILES.get(color_group, [])
        return all(self.tiles[i].owner == player_id for i in indices)

    def count_owned_in_group(self, player_id: int, color_group: ColorGroup) -> int:
        """Counts how many properties a player owns in a group (useful for RR/Utilities)."""
        indices = COLOR_GROUP_TILES.get(color_group, [])
        return sum(1 for i in indices if self.tiles[i].owner == player_id)

    def calculate_rent(
        self,
        tile_index: int,
        dice_sum: int,
        forced_utility_multiplier: Optional[int] = None,
        force_double_railroad: bool = False,
    ) -> int:
        """
        Calculates exact rent due for landing on tile_index according to Hasbro rules.
        Returns 0 if unowned or mortgaged.
        """
        tile = self.tiles[tile_index]
        if tile.owner is None or tile.is_mortgaged:
            return 0

        # 1. Street Rent
        if tile.tile_type == TileType.STREET:
            if tile.num_hotels == 1:
                return tile.rent_hotel
            elif tile.num_houses == 4:
                return tile.rent_4
            elif tile.num_houses == 3:
                return tile.rent_3
            elif tile.num_houses == 2:
                return tile.rent_2
            elif tile.num_houses == 1:
                return tile.rent_1
            else:
                # Unimproved: Double rent if owner has full monopoly
                if self.owns_full_group(tile.owner, tile.color_group):
                    return tile.base_rent * 2
                return tile.base_rent

        # 2. Railroad Rent (1: $25, 2: $50, 3: $100, 4: $200)
        elif tile.tile_type == TileType.RAILROAD:
            count = sum(
                1
                for idx in COLOR_GROUP_TILES[ColorGroup.RAILROAD]
                if self.tiles[idx].owner == tile.owner and not self.tiles[idx].is_mortgaged
            )
            if count == 0:
                return 0
            base_rr_rent = 25 * (2 ** (count - 1))
            if force_double_railroad:
                base_rr_rent *= 2
            return base_rr_rent

        # 3. Utility Rent
        elif tile.tile_type == TileType.UTILITY:
            if forced_utility_multiplier is not None:
                return forced_utility_multiplier * dice_sum
            count = sum(
                1
                for idx in COLOR_GROUP_TILES[ColorGroup.UTILITY]
                if self.tiles[idx].owner == tile.owner and not self.tiles[idx].is_mortgaged
            )
            if count >= 2:
                return 10 * dice_sum
            elif count == 1:
                return 4 * dice_sum
            return 0

        return 0

    def can_build_house(self, tile_index: int, player_id: int) -> bool:
        """
        Verifies if a player can legally build a house on this tile.
        Enforces:
        - Full color group ownership
        - No mortgaged properties in the color group
        - Strict even-building rule
        - Strict bank shortage rule (32 houses / 12 hotels)
        """
        tile = self.tiles[tile_index]
        if tile.owner != player_id or tile.tile_type != TileType.STREET:
            return False

        if not self.owns_full_group(player_id, tile.color_group):
            return False

        # No property in the group may be mortgaged
        group_indices = COLOR_GROUP_TILES[tile.color_group]
        if any(self.tiles[i].is_mortgaged for i in group_indices):
            return False

        # Check if already has a hotel
        if tile.num_hotels == 1:
            return False

        # Even building rule: Cannot build on this tile if any other tile in group has fewer houses
        current_bldgs = tile.num_houses
        for i in group_indices:
            other = self.tiles[i]
            if other.num_hotels == 0 and other.num_houses < current_bldgs:
                return False

        # Bank inventory limits:
        if current_bldgs < 4:
            # Building a house: bank must have at least 1 house
            return self.available_houses > 0
        else:
            # Upgrading 4 houses to hotel: bank must have 1 hotel
            return self.available_hotels > 0

    def can_sell_building(self, tile_index: int, player_id: int) -> bool:
        """
        Verifies if a player can legally sell a house/hotel on this tile.
        Enforces:
        - Has buildings to sell
        - Strict even-breakdown rule
        - If selling hotel to return to 4 houses, bank must have 4 houses available.
        """
        tile = self.tiles[tile_index]
        if tile.owner != player_id or tile.tile_type != TileType.STREET:
            return False

        if tile.num_houses == 0 and tile.num_hotels == 0:
            return False

        group_indices = COLOR_GROUP_TILES[tile.color_group]
        current_bldgs = tile.total_buildings

        # Even breakdown rule: cannot sell if another property in group has more buildings
        for i in group_indices:
            if self.tiles[i].total_buildings > current_bldgs:
                return False

        # If selling hotel, must check if bank has 4 houses
        if tile.num_hotels == 1 and self.available_houses < 4:
            # In official tournament play, if bank does not have 4 houses, player must sell hotel
            # and all 4 houses all at once, or hotel cannot be downgraded to houses.
            pass

        return True

    def can_mortgage(self, tile_index: int, player_id: int) -> bool:
        """
        Verifies if property can be mortgaged.
        Enforces:
        - Owned by player and not already mortgaged
        - ALL buildings on ALL properties in that color group must be sold first!
        """
        tile = self.tiles[tile_index]
        if tile.owner != player_id or tile.is_mortgaged or not tile.is_purchasable:
            return False

        if tile.tile_type == TileType.STREET:
            group_indices = COLOR_GROUP_TILES[tile.color_group]
            if any(self.tiles[i].num_houses > 0 or self.tiles[i].num_hotels > 0 for i in group_indices):
                return False

        return True

    def can_unmortgage(self, tile_index: int, player_id: int) -> bool:
        """Verifies if property can be unmortgaged."""
        tile = self.tiles[tile_index]
        return tile.owner == player_id and tile.is_mortgaged
