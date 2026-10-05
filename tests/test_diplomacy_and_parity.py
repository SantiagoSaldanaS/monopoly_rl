"""
Unit and integration tests for Action Parity (single house build/sell)
and Diplomatic Alliances (rent truces & multi-asset bargaining).
"""

import unittest
from monopoly_core import MonopolyGame, ColorGroup, MAX_HOUSES, MAX_HOTELS
from play_server import GameSession, BuildRequest, PropRequest, ChatRequest, NegotiateRequest


class TestDiplomacyAndParity(unittest.TestCase):

    def setUp(self):
        self.session = GameSession(seed=123)
        self.game = self.session.game
        self.board = self.game.board
        self.players = self.game.players

    def test_single_house_build_and_sell_rules(self):
        """Test single-house build and breakdown enforces official Hasbro even-building rules."""
        # Player 0 owns Boardwalk (39) & Park Place (37)
        self.board.tiles[37].owner = 0
        self.board.tiles[39].owner = 0

        # Cannot build 2 on Park Place before Boardwalk has 1
        self.assertTrue(self.board.can_build_house(37, 0))
        self.assertTrue(self.board.can_build_house(39, 0))

        # Build 1 house on Park Place
        res = self.session.build_house(37)
        self.assertEqual(self.board.tiles[37].num_houses, 1)
        self.assertEqual(self.board.tiles[39].num_houses, 0)

        # Now cannot build another house on Park Place until Boardwalk has 1
        self.assertFalse(self.board.can_build_house(37, 0))
        self.assertTrue(self.board.can_build_house(39, 0))

        # Build 1 house on Boardwalk
        self.session.build_house(39)
        self.assertEqual(self.board.tiles[39].num_houses, 1)

        # Now can sell 1 house from Park Place (refund 50% = $100)
        cash_before = self.players[0].cash
        self.assertTrue(self.board.can_sell_building(37, 0))
        self.session.sell_house(37)
        self.assertEqual(self.board.tiles[37].num_houses, 0)
        self.assertEqual(self.players[0].cash, cash_before + 100)

    def test_sell_tier(self):
        """Test sell_tier sells 1 house across all properties in color group."""
        self.board.tiles[1].owner = 0
        self.board.tiles[3].owner = 0
        self.board.tiles[1].num_houses = 2
        self.board.tiles[3].num_houses = 2

        cash_before = self.players[0].cash
        self.session.sell_tier("BROWN")
        self.assertEqual(self.board.tiles[1].num_houses, 1)
        self.assertEqual(self.board.tiles[3].num_houses, 1)
        # Brown house cost is $50, resale is $25 each -> $50 total
        self.assertEqual(self.players[0].cash, cash_before + 50)

    def test_rent_truce_immunity(self):
        """Test diplomatic treaty waives rent when landing on bot's property."""
        # Bot 1 (PPO) owns Boardwalk with a hotel
        self.board.tiles[39].owner = 1
        self.board.tiles[39].num_hotels = 1

        # Without treaty, landing on Boardwalk incurs $2000 rent
        self.players[0].position = 39
        p0_cash_before = self.players[0].cash

        # Register a rent truce between Player 0 and Bot 1
        self.session.active_treaties.append({
            "id": "truce1",
            "type": "RENT_TRUCE",
            "party_a": 0,
            "party_b": 1,
            "title": "Truce: Player 1 & PPO",
            "description": "Waives rent between Player 1 and PPO",
            "turns_remaining": 10,
            "created_turn": 0,
        })

        # Resolve landing
        self.game.resolve_tile_landing(self.players[0])

        # Rent should be waived! Cash should be unchanged
        self.assertEqual(self.players[0].cash, p0_cash_before)

    def test_trade_negotiation_properties_and_cash(self):
        """Test that trade negotiation allows bargaining for deeds as well as cash."""
        # Bot 1 wants Mediterranean Avenue (1) from Player 0
        self.board.tiles[1].owner = 0
        self.board.tiles[3].owner = 1  # Bot 1 owns Baltic Avenue (3)

        self.session.incoming_trade_proposal = {
            "bot_id": 1,
            "bot_name": "Politically Perfect Organism",
            "bot_wants_props": [1],
            "bot_wants_props_names": ["Mediterranean Avenue"],
            "bot_gives_props": [],
            "bot_gives_props_names": [],
            "bot_gives_cash": 40,
            "max_cash_willing": 150,
            "proposal_message": "Let us trade.",
        }

        # Bargain for Baltic Avenue
        res = self.session.negotiate_trade("Give me Baltic Avenue and we have a deal")
        self.assertIn("Baltic Avenue", res["response"])
        self.assertIn(3, res["new_gives_props"])
        self.assertIn("Baltic Avenue", res["new_gives_props_names"])

        # Bargain for cash
        res2 = self.session.negotiate_trade("I also need $100 cash")
        self.assertEqual(res2["new_cash"], 100)

    def test_chat_alliance_ratification(self):
        """Test chat endpoint creates real RENT_TRUCE treaties in active_treaties."""
        res = self.session.chat(
            bot_id=1,
            message="Let's form an alliance and agree to a rent truce for 15 turns"
        )
        self.assertTrue(res["treaty_ratified"])
        self.assertGreater(len(self.session.active_treaties), 0)
        treaty = self.session.active_treaties[-1]
        self.assertEqual(treaty["type"], "RENT_TRUCE")
        self.assertEqual(treaty["party_a"], 0)
        self.assertEqual(treaty["party_b"], 1)
        self.assertGreater(treaty["turns_remaining"], 0)


if __name__ == "__main__":
    unittest.main()
