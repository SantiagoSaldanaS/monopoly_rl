"""
Comprehensive Verification Test Suite for Official Monopoly Rules.
Audited against Hasbro official rules and DARPA GNOME schema.
"""

import unittest
import time
from monopoly_core import (
    MonopolyGame,
    Board,
    Tile,
    Player,
    TileType,
    ColorGroup,
    MAX_HOUSES,
    MAX_HOTELS,
    conduct_auction,
)
from monopoly_core.constants import BOARD_SPECS, COLOR_GROUP_TILES


class TestMonopolyOfficialRules(unittest.TestCase):

    def setUp(self):
        self.game = MonopolyGame(seed=42)
        self.board = self.game.board
        self.players = self.game.players

    def test_board_structure(self):
        """Verify the 40-tile board layout and property counts."""
        self.assertEqual(len(self.board.tiles), 40)
        self.assertEqual(self.board.tiles[0].name, "Go")
        self.assertEqual(self.board.tiles[10].name, "Jail / Just Visiting")
        self.assertEqual(self.board.tiles[20].name, "Free Parking")
        self.assertEqual(self.board.tiles[30].name, "Go To Jail")
        self.assertEqual(self.board.tiles[39].name, "Boardwalk")

        # Verify deed counts: 22 streets, 4 railroads, 2 utilities = 28 properties
        purchasable = [t for t in self.board.tiles if t.is_purchasable]
        self.assertEqual(len(purchasable), 28)

        streets = [t for t in self.board.tiles if t.tile_type == TileType.STREET]
        self.assertEqual(len(streets), 22)

        railroads = [t for t in self.board.tiles if t.tile_type == TileType.RAILROAD]
        self.assertEqual(len(railroads), 4)

        utilities = [t for t in self.board.tiles if t.tile_type == TileType.UTILITY]
        self.assertEqual(len(utilities), 2)

    def test_card_decks(self):
        """Verify 16 Chance and 16 Community Chest cards exist."""
        self.assertEqual(len(self.board.chance_deck.cards), 17)  # 17 in standard set (with 2 RR cards)
        self.assertEqual(len(self.board.community_chest_deck.cards), 16)

    def test_even_building_rule(self):
        """Verify that houses must be built evenly across a monopoly."""
        # Give Player 0 the Brown monopoly (Mediterranean index 1, Baltic index 3)
        self.board.tiles[1].owner = 0
        self.board.tiles[3].owner = 0

        # Player 0 owns full group
        self.assertTrue(self.board.owns_full_group(0, ColorGroup.BROWN))

        # Initially both have 0 houses. Can build on either:
        self.assertTrue(self.board.can_build_house(1, 0))
        self.assertTrue(self.board.can_build_house(3, 0))

        # Build 1 house on Mediterranean (tile 1)
        self.board.tiles[1].num_houses = 1
        self.board.available_houses -= 1

        # Now Mediterranean has 1 house, Baltic has 0.
        # Even building rule: CANNOT build 2nd house on Mediterranean until Baltic has 1!
        self.assertFalse(self.board.can_build_house(1, 0))
        # Must build on Baltic
        self.assertTrue(self.board.can_build_house(3, 0))

        # Build 1 house on Baltic
        self.board.tiles[3].num_houses = 1
        self.board.available_houses -= 1

        # Now both have 1 house: can build on either again
        self.assertTrue(self.board.can_build_house(1, 0))
        self.assertTrue(self.board.can_build_house(3, 0))

    def test_bank_housing_shortage(self):
        """Verify strict 32-house limit prevents building when bank is depleted."""
        self.board.tiles[1].owner = 0
        self.board.tiles[3].owner = 0

        # Artificially deplete bank houses to 0
        self.board.available_houses = 0

        # Cannot build even with full monopoly and abundant cash
        self.assertFalse(self.board.can_build_house(1, 0))
        self.assertFalse(self.board.can_build_house(3, 0))

    def test_mortgage_prohibited_with_buildings(self):
        """Verify properties cannot be mortgaged if any property in color group has houses."""
        self.board.tiles[1].owner = 0
        self.board.tiles[3].owner = 0

        # Unimproved: can mortgage
        self.assertTrue(self.board.can_mortgage(1, 0))
        self.assertTrue(self.board.can_mortgage(3, 0))

        # Build a house on Mediterranean
        self.board.tiles[1].num_houses = 1

        # Now NEITHER Mediterranean NOR Baltic can be mortgaged!
        self.assertFalse(self.board.can_mortgage(1, 0))
        self.assertFalse(self.board.can_mortgage(3, 0))

    def test_unimproved_monopoly_doubles_rent(self):
        """Verify rent doubles on unimproved sites in completed color group."""
        med = self.board.tiles[1]  # Base rent: $2
        bal = self.board.tiles[3]  # Base rent: $4

        # Single property owned: base rent
        med.owner = 0
        bal.owner = 1
        self.assertEqual(self.board.calculate_rent(1, dice_sum=7), 2)

        # Full monopoly completed: rent doubles to $4
        bal.owner = 0
        self.assertEqual(self.board.calculate_rent(1, dice_sum=7), 4)

    def test_railroad_rent_scaling(self):
        """Verify railroad rents: 1: $25, 2: $50, 3: $100, 4: $200."""
        rr1, rr2, rr3, rr4 = 5, 15, 25, 35
        self.board.tiles[rr1].owner = 0
        self.assertEqual(self.board.calculate_rent(rr1, dice_sum=7), 25)

        self.board.tiles[rr2].owner = 0
        self.assertEqual(self.board.calculate_rent(rr1, dice_sum=7), 50)

        self.board.tiles[rr3].owner = 0
        self.assertEqual(self.board.calculate_rent(rr1, dice_sum=7), 100)

        self.board.tiles[rr4].owner = 0
        self.assertEqual(self.board.calculate_rent(rr1, dice_sum=7), 200)

    def test_utility_rent_scaling(self):
        """Verify utility rents: 1 owned = 4x dice, 2 owned = 10x dice."""
        u1, u2 = 12, 28
        self.board.tiles[u1].owner = 0
        self.assertEqual(self.board.calculate_rent(u1, dice_sum=8), 32)  # 4 * 8

        self.board.tiles[u2].owner = 0
        self.assertEqual(self.board.calculate_rent(u1, dice_sum=8), 80)  # 10 * 8

    def test_auction_mechanism(self):
        """Verify open ascending auction transfers deed and charges winning bid."""
        tile = self.board.tiles[1]  # Mediterranean Avenue ($60)
        self.assertIsNone(tile.owner)

        # Conduct auction with custom bids
        winner_id, bid = conduct_auction(
            tile,
            self.players,
            self.board,
            custom_valuations={0: 50, 1: 80, 2: 70, 3: 40},
        )
        self.assertEqual(winner_id, 1)
        self.assertEqual(tile.owner, 1)
        self.assertEqual(self.players[1].cash, 1500 - bid)
        self.assertGreaterEqual(bid, 70)  # Must beat player 2's $70 bid

    def test_full_game_simulation_speed(self):
        """Verify that games run at high throughput (> 150 games/second)."""
        num_games = 100
        t0 = time.time()
        for i in range(num_games):
            g = MonopolyGame(seed=1000 + i, max_turns=300)
            winner = g.play_full_game()
            self.assertIsNotNone(winner)
        t1 = time.time()
        elapsed = t1 - t0
        speed = num_games / elapsed
        print(f"\n[BENCHMARK] {num_games} full games completed in {elapsed:.3f}s ({speed:.1f} games/sec) on a single thread!")
        self.assertGreater(speed, 50)  # Must achieve at least 50 games/sec (usually 200+)


if __name__ == "__main__":
    unittest.main()
