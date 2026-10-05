"""
RL Agent Wrapper for MaskedActorCritic Model.
Allows trained PyTorch policies to play inside MonopolyGame and Tournament harnesses.
"""

from typing import TYPE_CHECKING
import torch
import numpy as np

from monopoly_core.constants import ALL_PROPERTY_INDICES, COLOR_GROUP_TILES, ColorGroup, JAIL_FINE
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
from monopoly_env.obs_encoder import encode_observation
from models.actor_critic import MaskedActorCritic

if TYPE_CHECKING:
    from monopoly_core.game import MonopolyGame
    from monopoly_core.player import Player
    from monopoly_core.board import Tile


class RLAgent:
    def __init__(self, model_path: str = "checkpoints/monopoly_ppo_final.pt", device: str = "cpu", name: str = "PPO-Agent", deterministic: bool = True):
        self.name = name
        self.device = device
        self.deterministic = deterministic
        self.model = MaskedActorCritic(obs_dim=439, act_dim=TOTAL_ACTIONS).to(device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.eval()
        except Exception as e:
            print(f"[WARN] Could not load model from '{model_path}': {e}")

    def on_buy_decision(self, player: "Player", tile: "Tile", game: "MonopolyGame") -> bool:
        """Determines whether to purchase the unowned property or send to auction via trained policy."""
        obs = encode_observation(game, player.player_id)
        mask = get_action_mask(game, player.player_id, pending_decision="BUY_OR_AUCTION")
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            dist, _ = self.model(obs_t, mask_t)
            act = torch.argmax(dist.logits).item() if self.deterministic else dist.sample().item()

        return act == Action.BUY_PROPERTY

    def on_pre_roll(self, player: "Player", game: "MonopolyGame"):
        """Executes strategic house building, selling, mortgaging, or unmortgaging via trained policy."""
        modified_props = set()
        for _ in range(10):  # Allow up to 10 policy actions per turn
            obs = encode_observation(game, player.player_id)
            mask = get_action_mask(game, player.player_id, pending_decision="MANAGEMENT", modified_props=modified_props)

            if mask.sum() <= 1 and mask[Action.END_OR_ROLL] == 1:
                break

            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)

            with torch.no_grad():
                dist, _ = self.model(obs_t, mask_t)
                act = torch.argmax(dist.logits).item() if self.deterministic else dist.sample().item()

            if act == Action.END_OR_ROLL:
                break

            # Execute group-level tier building
            if BUILD_GROUP_START <= act < SELL_GROUP_START:
                g_idx = act - BUILD_GROUP_START
                group = BUILDABLE_GROUPS[g_idx]
                indices = COLOR_GROUP_TILES[group]
                for _ in range(len(indices)):
                    min_dev = min(game.board.tiles[i].num_houses + (5 if game.board.tiles[i].num_hotels else 0) for i in indices)
                    if min_dev >= 5:
                        break
                    built = False
                    for i in indices:
                        dev = game.board.tiles[i].num_houses + (5 if game.board.tiles[i].num_hotels else 0)
                        if dev == min_dev and game.board.can_build_house(i, player.player_id):
                            tile = game.board.tiles[i]
                            if player.cash >= tile.house_cost:
                                player.cash -= tile.house_cost
                                if tile.num_houses == 4:
                                    tile.num_houses = 0
                                    tile.num_hotels = 1
                                    game.board.available_houses += 4
                                    game.board.available_hotels -= 1
                                else:
                                    tile.num_houses += 1
                                    game.board.available_houses -= 1
                                built = True
                                break
                    if not built:
                        break

            # Execute group-level tier selling
            elif SELL_GROUP_START <= act < MORTGAGE_START:
                g_idx = act - SELL_GROUP_START
                group = BUILDABLE_GROUPS[g_idx]
                indices = COLOR_GROUP_TILES[group]
                for _ in range(len(indices)):
                    max_dev = max(game.board.tiles[i].num_houses + (5 if game.board.tiles[i].num_hotels else 0) for i in indices)
                    if max_dev <= 0:
                        break
                    sold = False
                    for i in reversed(indices):
                        dev = game.board.tiles[i].num_houses + (5 if game.board.tiles[i].num_hotels else 0)
                        if dev == max_dev:
                            tile = game.board.tiles[i]
                            sell_val = tile.house_cost // 2
                            if tile.num_hotels == 1:
                                if game.board.available_houses >= 4:
                                    tile.num_hotels = 0
                                    tile.num_houses = 4
                                    game.board.available_hotels += 1
                                    game.board.available_houses -= 4
                                    player.cash += sell_val
                                else:
                                    tile.num_hotels = 0
                                    tile.num_houses = 0
                                    game.board.available_hotels += 1
                                    player.cash += sell_val * 5
                            else:
                                tile.num_houses -= 1
                                game.board.available_houses += 1
                                player.cash += sell_val
                            sold = True
                            break
                    if not sold:
                        break

            # Execute mortgaging
            elif MORTGAGE_START <= act < UNMORTGAGE_START:
                prop_idx = ALL_PROPERTY_INDICES[act - MORTGAGE_START]
                modified_props.add(prop_idx)
                tile = game.board.tiles[prop_idx]
                if game.board.can_mortgage(prop_idx, player.player_id):
                    tile.is_mortgaged = True
                    player.cash += tile.mortgage_value

            # Execute unmortgaging
            elif UNMORTGAGE_START <= act < TOTAL_ACTIONS:
                prop_idx = ALL_PROPERTY_INDICES[act - UNMORTGAGE_START]
                modified_props.add(prop_idx)
                tile = game.board.tiles[prop_idx]
                cost = int(tile.mortgage_value * 1.1)
                if player.cash >= cost and game.board.can_unmortgage(prop_idx, player.player_id):
                    player.cash -= cost
                    tile.is_mortgaged = False

    def on_jail_decision(self, player: "Player", game: "MonopolyGame"):
        """Decides whether to pay the fine, use card, or attempt doubles."""
        obs = encode_observation(game, player.player_id)
        mask = get_action_mask(game, player.player_id, pending_decision="JAIL")
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            dist, _ = self.model(obs_t, mask_t)
            act = torch.argmax(dist.logits).item() if self.deterministic else dist.sample().item()

        if act == Action.USE_JAIL_CARD and player.get_out_of_jail_cards > 0:
            player.get_out_of_jail_cards -= 1
            player.in_jail = False
            player.jail_turns = 0
        elif act == Action.PAY_JAIL_FINE and player.cash >= JAIL_FINE:
            player.cash -= JAIL_FINE
            player.in_jail = False
            player.jail_turns = 0
