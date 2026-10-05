"""
Gymnasium / PettingZoo Compatible Environment for Official Monopoly Rules.
Features strategic dense rewards and automated opponent turn resolution.
"""

from typing import Optional, Any
import numpy as np

from monopoly_core.game import MonopolyGame
from monopoly_core.constants import ALL_PROPERTY_INDICES, COLOR_GROUP_TILES, JAIL_FINE, ColorGroup
from monopoly_core.board import TileType
from monopoly_core.auction import conduct_auction
from monopoly_env.obs_encoder import encode_observation
from monopoly_env.action_space import (
    Action,
    TOTAL_ACTIONS,
    BUILD_GROUP_START,
    SELL_GROUP_START,
    MORTGAGE_START,
    UNMORTGAGE_START,
    BUILDABLE_GROUPS,
    get_action_mask,
)
class MonopolyEnv:
    """
    Environment where Player 0 is the learning RL agent and Players 1, 2, 3
    are autonomous league opponents (Markov-ROI, Aggressive, Conservative).
    """

    def __init__(self, max_turns: int = 400, seed: Optional[int] = None, opponents: Optional[list] = None):
        self.max_turns = max_turns
        self.seed = seed
        self.agents = [f"player_{i}" for i in range(4)]
        if opponents is None:
            from agents.markov_agent import MarkovROIAgent
            from agents.baseline_agents import AggressiveAgent, ConservativeAgent
            opponents = [MarkovROIAgent(), AggressiveAgent(), ConservativeAgent()]
        self.default_opponents = opponents
        self.game = MonopolyGame(num_players=4, seed=seed, max_turns=max_turns)
        self._attach_opponents()
        self.pending_decision = "MANAGEMENT"
        self.current_agent_idx = 0
        self.turn_modified_props: set[int] = set()

    def _attach_opponents(self):
        # Player 0 has no heuristic agent (controlled by RL)
        self.game.players[0].agent = None
        for i, opp in enumerate(self.default_opponents):
            self.game.players[i + 1].agent = opp

    def reset(self, seed: Optional[int] = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            self.seed = seed
        self.game = MonopolyGame(num_players=4, seed=self.seed, max_turns=self.max_turns)
        self._attach_opponents()
        self.current_agent_idx = 0
        self.pending_decision = "MANAGEMENT"
        self.turn_modified_props = set()

        obs = {"player_0": encode_observation(self.game, 0)}
        info = {"player_0": {"action_mask": get_action_mask(self.game, 0, self.pending_decision, self.turn_modified_props)}}
        return obs, info

    def _compute_potential(self, player_id: int) -> float:
        player = self.game.players[player_id]
        if player.is_bankrupt:
            return -10.0
        board = self.game.board
        nw = player.net_worth(board) / 500.0
        props = sum(1 for t in board.tiles if t.owner == player_id) * 0.35

        # Monopolies: ready-to-build monopolies are worth 2.0 each
        monopolies = 0.0
        for g in BUILDABLE_GROUPS:
            if board.owns_full_group(player_id, g):
                indices = COLOR_GROUP_TILES[g]
                # Fully unmortgaged monopoly -> full value!
                if not any(board.tiles[i].is_mortgaged for i in indices):
                    monopolies += 2.0
                else:
                    monopolies += 0.8  # Impaired by mortgage

        houses = 0.0
        for t in board.tiles:
            if t.owner == player_id:
                houses += t.total_buildings * 0.85
                if t.num_houses >= 3 or t.num_hotels >= 1:
                    houses += 0.50  # Sweet-spot 3-house bonus

        mortgages = sum(1 for t in board.tiles if t.owner == player_id and t.is_mortgaged) * 0.25
        return float(nw + props + (monopolies * 2.0) + houses - mortgages)

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, bool], dict[str, bool], dict[str, Any]]:
        player = self.game.players[0]
        mask = get_action_mask(self.game, 0, self.pending_decision, self.turn_modified_props)

        if mask[action] == 0:
            action = Action.END_OR_ROLL

        prev_potential = self._compute_potential(0)
        prev_bankrupt = player.is_bankrupt
        extra_reward = 0.0

        # 1. Decision handling for Player 0
        if self.pending_decision == "BUY_OR_AUCTION":
            tile = self.game.board.tiles[player.position]
            if action == Action.BUY_PROPERTY and player.cash >= tile.price:
                player.cash -= tile.price
                tile.owner = 0
                extra_reward += 0.30
                if tile.color_group is not None and self.game.board.owns_full_group(0, tile.color_group):
                    extra_reward += 2.0
            else:
                conduct_auction(tile, self.game.players, self.game.board)
                if player.cash >= tile.price + 150:
                    extra_reward -= 0.50
            self.pending_decision = "MANAGEMENT"
            self.turn_modified_props.clear()
            # Now advance to opponents' turns
            self._run_opponent_turns()

        elif action == Action.END_OR_ROLL:
            self.turn_modified_props.clear()
            landed_unowned = False
            if player.in_jail:
                # Roll doubles check
                d1, d2, is_double = self.game.roll_dice()
                dice_sum = d1 + d2
                if is_double:
                    player.in_jail = False
                    player.jail_turns = 0
                    player.position = (player.position + dice_sum) % 40
                    landed = self.game.board.tiles[player.position]
                    if landed.is_purchasable and landed.owner is None:
                        landed_unowned = True
                    else:
                        self.game.resolve_tile_landing(player, dice_sum)
                elif player.jail_turns >= 2:
                    if self.game.handle_payment(player, None, JAIL_FINE):
                        player.in_jail = False
                        player.jail_turns = 0
                        player.position = (player.position + dice_sum) % 40
                        landed = self.game.board.tiles[player.position]
                        if landed.is_purchasable and landed.owner is None:
                            landed_unowned = True
                        else:
                            self.game.resolve_tile_landing(player, dice_sum)
                else:
                    player.jail_turns += 1
            else:
                # Normal roll
                d1, d2, is_double = self.game.roll_dice()
                dice_sum = d1 + d2
                if is_double:
                    player.consecutive_doubles += 1
                    if player.consecutive_doubles >= 3:
                        self.game.send_to_jail(player)
                else:
                    player.consecutive_doubles = 0

                if not player.in_jail:
                    new_pos = (player.position + dice_sum) % 40
                    if new_pos < player.position:
                        player.cash += 200
                    player.position = new_pos
                    landed = self.game.board.tiles[player.position]
                    if landed.is_purchasable and landed.owner is None:
                        landed_unowned = True
                    else:
                        self.game.resolve_tile_landing(player, dice_sum)

            # If landed on unowned property -> pause for RL agent's BUY_OR_AUCTION choice
            if landed_unowned and not player.is_bankrupt and not self.game.game_over:
                self.pending_decision = "BUY_OR_AUCTION"
            else:
                self.pending_decision = "MANAGEMENT"
                self._run_opponent_turns()

        elif action == Action.PAY_JAIL_FINE and player.in_jail and player.cash >= JAIL_FINE:
            player.cash -= JAIL_FINE
            player.in_jail = False
            player.jail_turns = 0

        elif action == Action.USE_JAIL_CARD and player.in_jail and player.get_out_of_jail_cards > 0:
            player.get_out_of_jail_cards -= 1
            player.in_jail = False
            player.jail_turns = 0

        elif BUILD_GROUP_START <= action < SELL_GROUP_START:
            g_idx = action - BUILD_GROUP_START
            group = BUILDABLE_GROUPS[g_idx]
            indices = COLOR_GROUP_TILES[group]
            # Even tier building: advance all properties in the group evenly by 1 tier
            for _ in range(len(indices)):
                min_dev = min(self.game.board.tiles[i].num_houses + (5 if self.game.board.tiles[i].num_hotels else 0) for i in indices)
                if min_dev >= 5:
                    break
                built = False
                for i in indices:
                    dev = self.game.board.tiles[i].num_houses + (5 if self.game.board.tiles[i].num_hotels else 0)
                    if dev == min_dev and self.game.board.can_build_house(i, player.player_id):
                        tile = self.game.board.tiles[i]
                        if player.cash >= tile.house_cost:
                            player.cash -= tile.house_cost
                            if tile.num_houses == 4:
                                tile.num_houses = 0
                                tile.num_hotels = 1
                                self.game.board.available_houses += 4
                                self.game.board.available_hotels -= 1
                            else:
                                tile.num_houses += 1
                                self.game.board.available_houses -= 1
                            built = True
                            break
                if not built:
                    break

        elif SELL_GROUP_START <= action < MORTGAGE_START:
            g_idx = action - SELL_GROUP_START
            group = BUILDABLE_GROUPS[g_idx]
            indices = COLOR_GROUP_TILES[group]
            # Even tier selling: sell 1 tier across properties in the group evenly
            for _ in range(len(indices)):
                max_dev = max(self.game.board.tiles[i].num_houses + (5 if self.game.board.tiles[i].num_hotels else 0) for i in indices)
                if max_dev <= 0:
                    break
                sold = False
                for i in reversed(indices):
                    dev = self.game.board.tiles[i].num_houses + (5 if self.game.board.tiles[i].num_hotels else 0)
                    if dev == max_dev:
                        tile = self.game.board.tiles[i]
                        sell_val = tile.house_cost // 2
                        if tile.num_hotels == 1:
                            if self.game.board.available_houses >= 4:
                                tile.num_hotels = 0
                                tile.num_houses = 4
                                self.game.board.available_hotels += 1
                                self.game.board.available_houses -= 4
                                player.cash += sell_val
                            else:
                                tile.num_hotels = 0
                                tile.num_houses = 0
                                self.game.board.available_hotels += 1
                                player.cash += sell_val * 5
                        else:
                            tile.num_houses -= 1
                            self.game.board.available_houses += 1
                            player.cash += sell_val
                        sold = True
                        break
                if not sold:
                    break

        elif MORTGAGE_START <= action < UNMORTGAGE_START:
            prop_idx = ALL_PROPERTY_INDICES[action - MORTGAGE_START]
            self.turn_modified_props.add(prop_idx)
            tile = self.game.board.tiles[prop_idx]
            tile.is_mortgaged = True
            player.cash += tile.mortgage_value

        elif UNMORTGAGE_START <= action < TOTAL_ACTIONS:
            prop_idx = ALL_PROPERTY_INDICES[action - UNMORTGAGE_START]
            self.turn_modified_props.add(prop_idx)
            tile = self.game.board.tiles[prop_idx]
            cost = int(tile.mortgage_value * 1.1)
            player.cash -= cost
            tile.is_mortgaged = False

        # Potential-based reward: immune to cyclic reward-hacking loops
        curr_potential = self._compute_potential(0)
        step_reward = (curr_potential - prev_potential) + extra_reward

        if player.is_bankrupt and not prev_bankrupt:
            step_reward -= 5.0

        if self.game.game_over:
            if self.game.winner_id == 0:
                step_reward += 10.0  # Decisive tournament victory!
            else:
                step_reward -= 2.0

        rewards = {"player_0": float(step_reward)}
        terminated = {"player_0": self.game.game_over or player.is_bankrupt}
        truncated = {"player_0": False}

        obs = {"player_0": encode_observation(self.game, 0)}
        info = {"player_0": {"action_mask": get_action_mask(self.game, 0, self.pending_decision, self.turn_modified_props)}}

        return obs, rewards, terminated, truncated, info

    def _run_opponent_turns(self):
        """Runs opponents 1, 2, and 3 until it's Player 0's turn again or game over."""
        self.game._check_game_over()
        if self.game.game_over:
            return

        # Advance to Player 1
        self.game._advance_turn()

        # Execute turns for Players 1, 2, 3
        while self.game.current_player_idx != 0 and not self.game.game_over:
            opp_p = self.game.players[self.game.current_player_idx]
            if not opp_p.is_bankrupt:
                self.game.execute_turn()
            else:
                self.game._advance_turn()
            self.game._check_game_over()

        self.current_agent_idx = 0
