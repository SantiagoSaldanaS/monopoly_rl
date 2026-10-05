"""
Unit test for MonopolyEnv single-agent RL environment with automated league opponents.
"""

import unittest
import numpy as np
from monopoly_env import MonopolyEnv, TOTAL_ACTIONS, Action


class TestMonopolyEnv(unittest.TestCase):

    def test_env_reset(self):
        env = MonopolyEnv(seed=123)
        obs, info = env.reset()

        self.assertIn("player_0", obs)
        self.assertEqual(obs["player_0"].shape, (439,))
        self.assertEqual(obs["player_0"].dtype, np.float32)

        self.assertIn("action_mask", info["player_0"])
        mask = info["player_0"]["action_mask"]
        self.assertEqual(mask.shape, (TOTAL_ACTIONS,))
        self.assertEqual(mask[Action.END_OR_ROLL], 1)

    def test_env_stepping(self):
        env = MonopolyEnv(seed=456)
        obs, info = env.reset()

        for step_i in range(50):
            mask = info["player_0"]["action_mask"]
            legal_actions = np.where(mask == 1)[0]
            self.assertGreater(len(legal_actions), 0)

            action = np.random.choice(legal_actions)
            obs, rewards, term, trunc, info = env.step(action)

            self.assertIn("player_0", rewards)
            if term["player_0"]:
                break

    def test_env_speed(self):
        env = MonopolyEnv(seed=789)
        obs, info = env.reset()
        for _ in range(500):
            mask = info["player_0"]["action_mask"]
            legal_actions = np.where(mask == 1)[0]
            action = legal_actions[0]
            obs, rewards, term, trunc, info = env.step(action)
            if term["player_0"]:
                obs, info = env.reset()

    def test_group_level_building_and_selling(self):
        from monopoly_env.action_space import BUILD_GROUP_START, SELL_GROUP_START
        from monopoly_core.constants import ColorGroup
        env = MonopolyEnv(seed=42)
        env.reset()
        p0 = env.game.players[0]
        # Give Player 0 Orange monopoly (tiles 16, 18, 19) and abundant cash
        for t_idx in [16, 18, 19]:
            env.game.board.tiles[t_idx].owner = 0
        p0.cash = 2000

        # Group 3 is ORANGE (BROWN=0, SKY_BLUE=1, PINK=2, ORANGE=3)
        orange_build_act = BUILD_GROUP_START + 3
        orange_sell_act = SELL_GROUP_START + 3

        # Step 1: builds tier 1 across entire Orange group (1 house each)
        env.step(orange_build_act)
        self.assertEqual(env.game.board.tiles[16].num_houses, 1)
        self.assertEqual(env.game.board.tiles[18].num_houses, 1)
        self.assertEqual(env.game.board.tiles[19].num_houses, 1)

        # Step 2: builds tier 2 across entire Orange group (2 houses each)
        env.step(orange_build_act)
        self.assertEqual(env.game.board.tiles[16].num_houses, 2)
        self.assertEqual(env.game.board.tiles[18].num_houses, 2)
        self.assertEqual(env.game.board.tiles[19].num_houses, 2)

        # Step 3: sell tier 1 across entire Orange group (back to 1 house each)
        env.step(orange_sell_act)
        self.assertEqual(env.game.board.tiles[16].num_houses, 1)
        self.assertEqual(env.game.board.tiles[18].num_houses, 1)
        self.assertEqual(env.game.board.tiles[19].num_houses, 1)


if __name__ == "__main__":
    unittest.main()
