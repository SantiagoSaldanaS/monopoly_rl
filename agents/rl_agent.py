"""
RL Agent Wrapper for MaskedActorCritic Model.
Combines deep neural policy evaluation from overnight PPO training with
championship tournament instincts (monopoly completion, blocking, house starvation lock,
and late-game jail sanctuary) to achieve peak dominance across all active bots.
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
        """
        Determines whether to purchase the unowned property or send to auction.
        Combines deep neural policy evaluation with essential championship instincts:
        1. Always purchase if it completes our color group monopoly.
        2. Always purchase if it blocks an opponent from completing a monopoly.
        3. Evaluates general strategic value via trained neural policy.
        """
        cost = tile.price

        # 1. Absolute Priority: Complete our monopoly
        if tile.color_group is not None:
            indices = COLOR_GROUP_TILES.get(tile.color_group, [])
            my_count = sum(1 for i in indices if game.board.tiles[i].owner == player.player_id)
            if len(indices) > 0 and my_count == len(indices) - 1:
                return player.cash >= cost

        # 2. Defensive Priority: Block opponent near-monopoly
        if tile.color_group is not None:
            indices = COLOR_GROUP_TILES.get(tile.color_group, [])
            for opp in game.players:
                if opp.player_id != player.player_id and not opp.is_bankrupt:
                    opp_count = sum(1 for i in indices if game.board.tiles[i].owner == opp.player_id)
                    if len(indices) > 0 and opp_count == len(indices) - 1:
                        return player.cash >= cost

        # 3. Neural policy evaluation
        obs = encode_observation(game, player.player_id)
        mask = get_action_mask(game, player.player_id, pending_decision="BUY_OR_AUCTION")
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            dist, _ = self.model(obs_t, mask_t)
            act = torch.argmax(dist.logits).item() if self.deterministic else dist.sample().item()

        return act == Action.BUY_PROPERTY

    def on_pre_roll(self, player: "Player", game: "MonopolyGame"):
        """
        Executes high-velocity development and active asset management:
        - Priority order: Orange, Red, Sky Blue, Yellow, Pink, Dark Blue, Green, Brown.
        - House starvation lock: stops at 4 houses when bank <= 10 houses to choke opponents.
        - Preserves $100 emergency cash cushion.
        - Unmortgages high-traffic deeds when cash is healthy.
        """
        priority_groups = [
            ColorGroup.ORANGE,
            ColorGroup.RED,
            ColorGroup.SKY_BLUE,
            ColorGroup.YELLOW,
            ColorGroup.PINK,
            ColorGroup.DARK_BLUE,
            ColorGroup.GREEN,
            ColorGroup.BROWN,
        ]

        # 1. Building Phase
        for group in priority_groups:
            if not game.board.owns_full_group(player.player_id, group):
                continue
            indices = COLOR_GROUP_TILES.get(group, [])
            while True:
                built = False
                min_dev = min(game.board.tiles[i].num_houses + (5 if game.board.tiles[i].num_hotels else 0) for i in indices)
                # House starvation lock: refuse hotel upgrade if bank houses are scarce
                if min_dev == 4 and game.board.available_houses <= 10:
                    break
                if min_dev >= 5:
                    break
                for idx in indices:
                    tile = game.board.tiles[idx]
                    dev = tile.num_houses + (5 if tile.num_hotels else 0)
                    if dev == min_dev and game.board.can_build_house(idx, player.player_id):
                        if player.cash >= tile.house_cost + 100:
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

        # 2. Unmortgage Phase
        for idx in ALL_PROPERTY_INDICES:
            tile = game.board.tiles[idx]
            if tile.owner == player.player_id and tile.is_mortgaged:
                cost = int(tile.mortgage_value * 1.1)
                if player.cash >= cost + 120:
                    player.cash -= cost
                    tile.is_mortgaged = False

    def on_jail_decision(self, player: "Player", game: "MonopolyGame"):
        """
        Championship jail management:
        - Late game / active threats: If opponents own any buildings, remain in jail as long as possible (free safe sanctuary).
        - Early game: Exit immediately via card or fine to maximize board presence and acquire deeds.
        """
        opponents_have_buildings = any(
            t.total_buildings > 0
            for t in game.board.tiles
            if t.owner is not None and t.owner != player.player_id
        )
        if opponents_have_buildings:
            return  # Stay protected in jail

        # Early game / safe board
        if player.get_out_of_jail_cards > 0:
            player.get_out_of_jail_cards -= 1
            player.in_jail = False
            player.jail_turns = 0
        elif player.cash >= 200:
            player.cash -= JAIL_FINE
            player.in_jail = False
            player.jail_turns = 0

    def get_auction_bid(self, player: "Player", tile: "Tile", board: object) -> int:
        """
        Calculates strategic willingness to pay in open ascending auctions:
        - 2.5x printed price if property completes our monopoly.
        - 1.6x printed price if property blocks an opponent's monopoly.
        - 1.0x printed price for standard deeds.
        - Preserves $50 emergency cash cushion.
        """
        if player.is_bankrupt or player.cash <= 10:
            return 0

        max_affordable = max(0, player.cash - 50)
        if max_affordable <= 0:
            return 0

        valuation = tile.price
        group_indices = COLOR_GROUP_TILES.get(tile.color_group, [])
        if group_indices:
            unowned = [i for i in group_indices if i != tile.index and board.tiles[i].owner != player.player_id]
            if len(unowned) == 0:
                # Monopoly completion: top priority!
                valuation = int(tile.price * 2.5)
            else:
                for opp in range(4):
                    if opp != player.player_id:
                        opp_count = [i for i in group_indices if i != tile.index and board.tiles[i].owner == opp]
                        if len(opp_count) == len(group_indices) - 1:
                            # Block opponent near-monopoly!
                            valuation = max(valuation, int(tile.price * 1.6))

        return min(valuation, max_affordable)
