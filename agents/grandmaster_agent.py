"""
Tournament Grandmaster Heuristic Agent for Monopoly.
Implements championship meta-strategies:
1. The 32-House Bank Starvation Lock (refuses hotels when houses are scarce).
2. Dynamic Impending-Threat Cash Buffers (scales cash reserve by rents 2-12 tiles ahead).
3. Payback-Velocity Development (Orange/Red/Sky-Blue priority).
"""

from typing import TYPE_CHECKING
import numpy as np

from monopoly_core.constants import (
    ALL_PROPERTY_INDICES,
    BOARD_SPECS,
    COLOR_GROUP_TILES,
    ColorGroup,
    JAIL_FINE,
    TileType,
)
from agents.markov_agent import MARKOV_LANDING_PROBS

if TYPE_CHECKING:
    from monopoly_core.game import MonopolyGame
    from monopoly_core.player import Player
    from monopoly_core.board import Tile


class TournamentGrandmasterAgent:
    """
    World Championship-caliber heuristic agent.
    Combines Ash & Bishop Markov transition matrices with:
    - 32-house bank lock (holding houses to block opponents)
    - Dynamic forward threat scanning (cash reserve scales with danger zone)
    - High-velocity payback development
    """

    def __init__(self, name: str = "GrandmasterBot"):
        self.name = name

    def _calc_impending_threat(self, player: "Player", game: "MonopolyGame") -> int:
        """
        Calculates the maximum probable rent in the 2-12 roll window ahead.
        Roll 7 has 16.7% probability; 6 & 8 have 13.9%.
        """
        max_threat = 0
        for roll in range(2, 13):
            target_pos = (player.position + roll) % 40
            tile = game.board.tiles[target_pos]
            if tile.is_purchasable and tile.owner is not None and tile.owner != player.player_id and not tile.is_mortgaged:
                rent = game.board.calculate_rent(target_pos, roll)
                if rent > max_threat:
                    max_threat = rent
        return max_threat

    def _get_dynamic_cash_reserve(self, player: "Player", game: "MonopolyGame") -> int:
        """
        Computes dynamic cash reserve needed to survive the current danger zone.
        Base reserve is $120. If facing hotels in the 2-12 zone, reserves jump to 50% of max rent.
        """
        threat = self._calc_impending_threat(player, game)
        # In early game, $150 is plenty; in dangerous late game, reserve up to $600
        return max(150, min(600, int(threat * 0.6)))

    def on_buy_decision(self, player: "Player", tile: "Tile", game: "MonopolyGame") -> bool:
        """
        Buys properties that complete monopolies, block opponents, or have high Markov yield.
        """
        cost = tile.price
        reserve = self._get_dynamic_cash_reserve(player, game)

        # 1. ALWAYS buy if it completes OUR monopoly
        if tile.color_group is not None:
            group = tile.color_group
            indices = COLOR_GROUP_TILES.get(group, [])
            my_count = sum(1 for idx in indices if game.board.tiles[idx].owner == player.player_id)
            total_in_group = len(indices)
            if total_in_group > 0 and my_count == total_in_group - 1:
                return player.cash >= cost  # Buy at all costs!

        # 2. ALWAYS buy if it BLOCKS an opponent from completing a monopoly
        if tile.color_group is not None:
            indices = COLOR_GROUP_TILES.get(tile.color_group, [])
            for opp in game.players:
                if opp.player_id != player.player_id and not opp.is_bankrupt:
                    opp_count = sum(1 for idx in indices if game.board.tiles[idx].owner == opp.player_id)
                    if len(indices) > 0 and opp_count == len(indices) - 1:
                        # Opponent is 1 away from monopoly: BLOCK THEM!
                        return player.cash >= cost

        # 3. High-traffic properties (Orange, Red, Railroads, Sky Blue)
        prob = MARKOV_LANDING_PROBS.get(tile.index, 0.02)
        if prob >= 0.025 or tile.tile_type == TileType.RAILROAD:
            return player.cash >= cost + reserve

        # 4. Standard properties: buy if cash reserve is secure
        return player.cash >= cost + reserve + 100

    def on_pre_roll(self, player: "Player", game: "MonopolyGame"):
        """
        Grandmaster development:
        - Prioritizes 3 houses on Orange, Sky Blue, and Red (fastest breakeven).
        - HOUSE STARVATION: If bank has <= 10 houses, REFUSE hotels! Keep houses to choke opponents.
        """
        reserve = self._get_dynamic_cash_reserve(player, game)

        # Priority order for building by ROI / payback velocity
        priority_groups = [
            ColorGroup.ORANGE,    # Best landing frequency from Jail + fast payback
            ColorGroup.RED,       # High rent + high landing frequency
            ColorGroup.SKY_BLUE,  # Very cheap to hit 3 houses and starve bank
            ColorGroup.YELLOW,    # Solid mid-game
            ColorGroup.PINK,      # Cheap stepping stone
            ColorGroup.DARK_BLUE, # Devastating but expensive
            ColorGroup.GREEN,     # Slowest payback in the game
            ColorGroup.BROWN,     # Budget filler
        ]

        bank_houses = game.board.available_houses

        for group in priority_groups:
            if not game.board.owns_full_group(player.player_id, group):
                continue

            indices = COLOR_GROUP_TILES.get(group, [])

            # Build houses evenly
            while True:
                built_any = False
                min_houses = min(game.board.tiles[i].num_houses + (5 if game.board.tiles[i].num_hotels else 0) for i in indices)
                
                # Check house shortage strategy:
                # If we already have 4 houses, only upgrade to hotel if bank has plenty of houses (> 10)
                # If bank houses <= 10, STOP AT 4 HOUSES! (Starves the bank)
                if min_houses == 4 and bank_houses <= 10:
                    break

                if min_houses >= 5:
                    break

                for idx in indices:
                    tile = game.board.tiles[idx]
                    current_dev = tile.num_houses + (5 if tile.num_hotels else 0)
                    if current_dev == min_houses and game.board.can_build_house(idx, player.player_id):
                        if player.cash >= tile.house_cost + reserve:
                            player.cash -= tile.house_cost
                            if tile.num_houses == 4:
                                tile.num_houses = 0
                                tile.num_hotels = 1
                                game.board.available_houses += 4
                                game.board.available_hotels -= 1
                                bank_houses += 4
                            else:
                                tile.num_houses += 1
                                game.board.available_houses -= 1
                                bank_houses -= 1
                            built_any = True
                if not built_any:
                    break

        # Unmortgage properties if cash is healthy
        for idx in ALL_PROPERTY_INDICES:
            tile = game.board.tiles[idx]
            if tile.owner == player.player_id and tile.is_mortgaged:
                unmortgage_cost = int(tile.mortgage_value * 1.1)
                if player.cash >= unmortgage_cost + reserve + 150:
                    player.cash -= unmortgage_cost
                    tile.is_mortgaged = False

    def on_jail_decision(self, player: "Player", game: "MonopolyGame"):
        """
        Grandmaster jail strategy:
        - Early game (unowned properties remaining): Exit immediately (pay fine/use card) to buy deeds.
        - Late game (opponents own houses/hotels): STAY IN JAIL as long as possible! (Free safe haven).
        """
        # Count unowned properties
        unowned = sum(1 for t in game.board.tiles if t.is_purchasable and t.owner is None)

        if unowned > 6:
            # Early game: Pay fine or use card immediately to grab deeds
            if player.get_out_of_jail_cards > 0:
                player.get_out_of_jail_cards -= 1
                player.in_jail = False
                player.jail_turns = 0
            elif player.cash >= JAIL_FINE + 150:
                player.cash -= JAIL_FINE
                player.in_jail = False
                player.jail_turns = 0
        else:
            # Late game: Threat is high outside! Only roll doubles, don't pay until forced on turn 3!
            pass
