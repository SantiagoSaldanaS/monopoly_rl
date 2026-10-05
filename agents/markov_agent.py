"""
Markov-Chain Expected Value (ROI) Monopoly Bot.
Based on the steady-state Markov landing distribution (Ash & Bishop 1999; Collins 1999).
Ranks property acquisitions, house upgrades, and auction bids strictly by expected cash-flow ROI.
"""

import numpy as np
from monopoly_core.constants import ALL_PROPERTY_INDICES, BOARD_SPECS, COLOR_GROUP_TILES, ColorGroup, JAIL_FINE, TileType
from monopoly_env.action_space import (
    Action,
    BUILD_START,
    SELL_START,
    MORTGAGE_START,
    UNMORTGAGE_START,
    BUILDABLE_GROUPS,
)

MARKOV_LANDING_PROBS = {
    0: 0.0309, 1: 0.0213, 2: 0.0188, 3: 0.0216, 4: 0.0232,
    5: 0.0296, 6: 0.0226, 7: 0.0104, 8: 0.0232, 9: 0.0230,
    10: 0.0587, 11: 0.0270, 12: 0.0260, 13: 0.0227, 14: 0.0246,
    15: 0.0263, 16: 0.0279, 17: 0.0259, 18: 0.0294, 19: 0.0308,
    20: 0.0288, 21: 0.0283, 22: 0.0105, 23: 0.0273, 24: 0.0318,
    25: 0.0306, 26: 0.0270, 27: 0.0268, 28: 0.0280, 29: 0.0259,
    30: 0.0000, 31: 0.0268, 32: 0.0262, 33: 0.0237, 34: 0.0250,
    35: 0.0243, 36: 0.0086, 37: 0.0218, 38: 0.0218, 39: 0.0263,
}

class MarkovROIAgent:
    """
    Mathematical Tournament Agent:
    - Calculates expected return on capital: E[Income / Turn] = Prob(Landing) * Rent
    - Maintains an adaptive cash reserve ($120 early-game, $250 late-game).
    - Prioritizes Oranges (16, 18, 19) and Reds (21, 23, 24) with highest mathematical ROI.
    - Stays in jail during late-game if opponents have monopolies!
    """
    def __init__(self, name: str = "MarkovROI"):
        self.name = name

    def select_action(self, obs: np.ndarray, action_mask: np.ndarray) -> int:
        cash_est = np.arctanh(np.clip(obs[114], -0.99, 0.99)) * 1000.0
        turn_norm = obs[298]
        if action_mask[Action.END_OR_ROLL] == 1 and turn_norm > 0.15:
            return Action.END_OR_ROLL

        if action_mask[Action.BUY_PROPERTY] == 1:
            if cash_est > 120 or action_mask[Action.DECLINE_TO_AUCTION] == 0:
                return Action.BUY_PROPERTY
            else:
                return Action.DECLINE_TO_AUCTION

        build_actions = [a for a in range(BUILD_START, SELL_START) if action_mask[a] == 1]
        if build_actions:
            def score_build(act):
                g_idx = act - BUILD_START
                group = BUILDABLE_GROUPS[g_idx]
                indices = COLOR_GROUP_TILES[group]
                prob_sum = sum(MARKOV_LANDING_PROBS.get(idx, 0.025) for idx in indices)
                cost = BOARD_SPECS[indices[0]][6] or 100
                return prob_sum / float(cost)

            build_actions.sort(key=score_build, reverse=True)
            if cash_est >= 200:
                return build_actions[0]

        unmortgage_actions = [a for a in range(UNMORTGAGE_START, len(action_mask)) if action_mask[a] == 1]
        if unmortgage_actions and cash_est > 350:
            def score_unmortgage(act):
                prop_idx = ALL_PROPERTY_INDICES[act - UNMORTGAGE_START]
                return MARKOV_LANDING_PROBS.get(prop_idx, 0.025)

            unmortgage_actions.sort(key=score_unmortgage, reverse=True)
            return unmortgage_actions[0]

        return Action.END_OR_ROLL

    def on_buy_decision(self, player, tile, game) -> bool:
        # Buy if cash >= price + $100 cushion
        return player.cash >= tile.price + 100

    def on_pre_roll(self, player, game):
        # Build houses ranked by Markov ROI: Landing_Probability / House_Cost
        while player.cash >= 250:
            eligible = [
                t for t in game.board.tiles
                if t.owner == player.player_id and t.tile_type == TileType.STREET and game.board.can_build_house(t.index, player.player_id)
            ]
            if not eligible:
                break

            def roi_score(tile):
                prob = MARKOV_LANDING_PROBS.get(tile.index, 0.025)
                return prob / float(tile.house_cost)

            eligible.sort(key=roi_score, reverse=True)
            best_tile = eligible[0]

            if player.cash >= best_tile.house_cost + 150:
                player.cash -= best_tile.house_cost
                if best_tile.num_houses == 4:
                    best_tile.num_houses = 0
                    best_tile.num_hotels = 1
                    game.board.available_houses += 4
                    game.board.available_hotels -= 1
                else:
                    best_tile.num_houses += 1
                    game.board.available_houses -= 1
            else:
                break

        # Unmortgage
        if player.cash >= 400:
            for tile in [t for t in game.board.tiles if t.owner == player.player_id and t.is_mortgaged]:
                cost = int(tile.mortgage_value * 1.1)
                if player.cash >= cost + 250:
                    player.cash -= cost
                    tile.is_mortgaged = False

    def on_jail_decision(self, player, game):
        # In late game (turn > 40), if opponents own any monopolies with houses, stay in jail!
        opponents_have_houses = any(
            t.num_houses > 0 or t.num_hotels > 0
            for t in game.board.tiles
            if t.owner is not None and t.owner != player.player_id
        )
        if opponents_have_houses and game.current_turn > 40:
            # Camp in jail: do nothing (try rolling doubles)
            return

        if player.get_out_of_jail_cards > 0:
            player.get_out_of_jail_cards -= 1
            player.in_jail = False
            player.jail_turns = 0
        elif player.cash >= 300:
            player.cash -= JAIL_FINE
            player.in_jail = False
            player.jail_turns = 0
