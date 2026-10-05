"""
Baseline Heuristic Agents for Monopoly Benchmarking and League Play.
Includes discrete action selection and direct game-engine hook methods.
"""

import numpy as np
from monopoly_core.constants import (
    ALL_PROPERTY_INDICES,
    BOARD_SPECS,
    ColorGroup,
    JAIL_FINE,
    TileType,
)
from monopoly_env.action_space import (
    Action,
    BUILD_START,
    SELL_START,
    MORTGAGE_START,
    UNMORTGAGE_START,
)


class RandomAgent:
    """Selects uniformly at random among all currently legal actions."""
    def __init__(self, name: str = "Random"):
        self.name = name

    def select_action(self, obs: np.ndarray, action_mask: np.ndarray) -> int:
        legal_actions = np.where(action_mask > 0)[0]
        if len(legal_actions) == 0:
            return Action.END_OR_ROLL
        return int(np.random.choice(legal_actions))

    def on_buy_decision(self, player, tile, game) -> bool:
        return np.random.choice([True, False])

    def on_pre_roll(self, player, game):
        pass

    def on_jail_decision(self, player, game):
        pass


class AggressiveAgent:
    """
    Aggressive growth agent:
    - Prioritizes buying every property available.
    - Upgrades monopolies immediately.
    - Unmortgages properties as soon as affordable.
    - Exits jail immediately using fine or card.
    """
    def __init__(self, name: str = "Aggressive"):
        self.name = name

    def select_action(self, obs: np.ndarray, action_mask: np.ndarray) -> int:
        if action_mask[Action.BUY_PROPERTY] == 1:
            return Action.BUY_PROPERTY
        if action_mask[Action.USE_JAIL_CARD] == 1:
            return Action.USE_JAIL_CARD
        if action_mask[Action.PAY_JAIL_FINE] == 1:
            return Action.PAY_JAIL_FINE

        build_actions = [a for a in range(BUILD_START, SELL_START) if action_mask[a] == 1]
        if build_actions:
            return build_actions[0]

        unmortgage_actions = [a for a in range(UNMORTGAGE_START, len(action_mask)) if action_mask[a] == 1]
        if unmortgage_actions:
            return unmortgage_actions[0]

        return Action.END_OR_ROLL

    def on_buy_decision(self, player, tile, game) -> bool:
        return player.cash >= tile.price

    def on_pre_roll(self, player, game):
        # Build houses as long as cash >= house cost
        while player.cash >= 100:
            built_any = False
            for tile in sorted(
                [t for t in game.board.tiles if t.owner == player.player_id and t.tile_type == TileType.STREET],
                key=lambda t: t.house_cost,
            ):
                if game.board.can_build_house(tile.index, player.player_id) and player.cash >= tile.house_cost:
                    player.cash -= tile.house_cost
                    if tile.num_houses == 4:
                        tile.num_houses = 0
                        tile.num_hotels = 1
                        game.board.available_houses += 4
                        game.board.available_hotels -= 1
                    else:
                        tile.num_houses += 1
                        game.board.available_houses -= 1
                    built_any = True
                    break
            if not built_any:
                break

        # Unmortgage
        if player.cash >= 300:
            for tile in [t for t in game.board.tiles if t.owner == player.player_id and t.is_mortgaged]:
                cost = int(tile.mortgage_value * 1.1)
                if player.cash >= cost + 100:
                    player.cash -= cost
                    tile.is_mortgaged = False

    def on_jail_decision(self, player, game):
        if player.get_out_of_jail_cards > 0:
            player.get_out_of_jail_cards -= 1
            player.in_jail = False
            player.jail_turns = 0
        elif player.cash >= JAIL_FINE:
            player.cash -= JAIL_FINE
            player.in_jail = False
            player.jail_turns = 0


class ConservativeAgent:
    """
    Conservative defensive agent:
    - Only buys properties if cash buffer is high (> $400).
    - Hoards cash and remains in jail to avoid paying rent in late game.
    - Upgrades properties cautiously (only when cash > $600).
    """
    def __init__(self, name: str = "Conservative"):
        self.name = name

    def select_action(self, obs: np.ndarray, action_mask: np.ndarray) -> int:
        cash_norm = obs[114]
        if action_mask[Action.BUY_PROPERTY] == 1:
            if cash_norm > 0.4:  # roughly > $450
                return Action.BUY_PROPERTY
            elif action_mask[Action.DECLINE_TO_AUCTION] == 1:
                return Action.DECLINE_TO_AUCTION

        if action_mask[Action.END_OR_ROLL] == 1:
            return Action.END_OR_ROLL

        legal_actions = np.where(action_mask > 0)[0]
        return int(legal_actions[0])

    def on_buy_decision(self, player, tile, game) -> bool:
        # Requires $400 safety buffer
        return player.cash >= tile.price + 400

    def on_pre_roll(self, player, game):
        # Only build houses if cash > $600
        while player.cash >= 600:
            built_any = False
            for tile in sorted(
                [t for t in game.board.tiles if t.owner == player.player_id and t.tile_type == TileType.STREET],
                key=lambda t: t.house_cost,
            ):
                if game.board.can_build_house(tile.index, player.player_id) and player.cash >= tile.house_cost + 500:
                    player.cash -= tile.house_cost
                    if tile.num_houses == 4:
                        tile.num_houses = 0
                        tile.num_hotels = 1
                        game.board.available_houses += 4
                        game.board.available_hotels -= 1
                    else:
                        tile.num_houses += 1
                        game.board.available_houses -= 1
                    built_any = True
                    break
            if not built_any:
                break

    def on_jail_decision(self, player, game):
        # Prefer rolling doubles to stay safe in jail
        pass
