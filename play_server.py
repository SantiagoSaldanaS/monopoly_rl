"""
Interactive Monopoly Play Server.
Allows human players to play against the trained RL Agent, GrandmasterBot, and Markov-ROI
in a rich browser-based visual UI with live LLM diplomacy and spectator mode.
"""

import os
import random
import re
import sys
import webbrowser
from typing import Optional, Any
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

from monopoly_core.game import MonopolyGame
from monopoly_core.constants import (
    ALL_PROPERTY_INDICES,
    COLOR_GROUP_TILES,
    ColorGroup,
    JAIL_FINE,
    TileType,
)
from monopoly_core.auction import conduct_auction
from agents.rl_agent import RLAgent
from agents.grandmaster_agent import TournamentGrandmasterAgent
from agents.markov_agent import MarkovROIAgent
from agents.llm_strategist import LLMStrategist

app = FastAPI(title="Monopoly AI Arena")

# Mount visualizer directory for static assets
VISUALIZER_DIR = os.path.join(os.path.dirname(__file__), "visualizer")
os.makedirs(VISUALIZER_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=VISUALIZER_DIR), name="static")

BUILDABLE_GROUPS = [
    ColorGroup.BROWN,
    ColorGroup.SKY_BLUE,
    ColorGroup.PINK,
    ColorGroup.ORANGE,
    ColorGroup.RED,
    ColorGroup.YELLOW,
    ColorGroup.GREEN,
    ColorGroup.DARK_BLUE,
]


class GameSession:
    def __init__(self, seed: Optional[int] = None, model_path: str = "checkpoints/monopoly_ppo_final.pt"):
        self.seed = seed if (seed is not None and seed > 0) else random.randint(1, 99999999)
        self.model_path = model_path
        self.llm_strategist = LLMStrategist()
        self.player_skins = {0: 0, 1: 1, 2: 2, 3: 3}
        self.player_types = {0: "human", 1: "rl", 2: "grandmaster", 3: "markov"}
        self.player_personalities = {
            0: "Organic Lifeform",
            1: "Deep Policy Gradient",
            2: "Chess Grandmaster",
            3: "Stochastic Fatalist"
        }
        self.active_counter_offer: Optional[dict] = None
        self.latest_speech: Optional[dict] = None
        self.incoming_trade_proposal: Optional[dict] = None
        self.pending_auction: Optional[dict] = None
        self.last_human_tax_payment: Optional[dict] = None
        self.can_roll_again: bool = False
        self.bot_chatter_rate: str = "normal"
        self.recent_trades: list[tuple[int, int]] = []
        self.match_telemetry = {
            "turns": [],
            "cash": {0: [], 1: [], 2: [], 3: []},
            "net_worth": {0: [], 1: [], 2: [], 3: []},
            "props": {0: [], 1: [], 2: [], 3: []},
        }
        self.diplomatic_ledger = {
            1: {"affinity": 0, "gifts_received": [], "favors_owed": 0, "trades_completed": 0},
            2: {"affinity": 0, "gifts_received": [], "favors_owed": 0, "trades_completed": 0},
            3: {"affinity": 0, "gifts_received": [], "favors_owed": 0, "trades_completed": 0},
        }
        self.match_stats = {pid: {"cash_spent": 0, "properties_bought": 0, "houses_built": 0, "rent_paid": 0, "taxes_paid": 0, "mortgages_taken": 0, "trades_completed": 0} for pid in range(4)}
        self.reset(self.seed)

    def reset(self, seed: Optional[int] = None):
        if seed is not None and seed > 0:
            self.seed = seed
        else:
            self.seed = random.randint(1, 99999999)
        self.recent_trades = []
        self.pending_auction = None
        self.last_human_tax_payment = None
        self.game = MonopolyGame(num_players=4, seed=self.seed, max_turns=350, enable_logging=True)
        self.game.auction_handler = self._handle_game_auction
        self.game.interactive_human = True
        self.match_stats = {pid: {"cash_spent": 0, "properties_bought": 0, "houses_built": 0, "rent_paid": 0, "taxes_paid": 0, "mortgages_taken": 0, "trades_completed": 0} for pid in range(4)}
        self.diplomatic_ledger = {
            1: {"affinity": 0, "gifts_received": [], "favors_owed": 0, "trades_completed": 0},
            2: {"affinity": 0, "gifts_received": [], "favors_owed": 0, "trades_completed": 0},
            3: {"affinity": 0, "gifts_received": [], "favors_owed": 0, "trades_completed": 0},
        }
        self.match_telemetry = {
            "turns": [],
            "cash": {0: [], 1: [], 2: [], 3: []},
            "net_worth": {0: [], 1: [], 2: [], 3: []},
            "props": {0: [], 1: [], 2: [], 3: []},
        }

        # Player 0 is Human (Player 1)
        self.game.players[0].name = "Player 1"
        self.game.players[0].agent = None

        # Player 1 is Politically Perfect Organism (RL Agent)
        self.game.players[1].name = "Politically Perfect Organism"
        try:
            self.game.players[1].agent = RLAgent(model_path=self.model_path, device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") != "" else "cpu")
        except Exception:
            self.game.players[1].agent = RLAgent(model_path=self.model_path, device="cpu")

        # Player 2 is Cash Elo (Tournament Grandmaster)
        self.game.players[2].name = "Cash Elo"
        self.game.players[2].agent = TournamentGrandmasterAgent()

        # Player 3 is Moneybags Markov (Markov-ROI Bot)
        self.game.players[3].name = "Moneybags Markov"
        self.game.players[3].agent = MarkovROIAgent()

        self.has_rolled = False
        self.can_roll_again = False
        self.pending_decision = "ROLL"
        self.last_dice = [0, 0]
        self.consecutive_doubles = 0
        self.active_counter_offer = None
        self.incoming_trade_proposal = None
        self.latest_speech = None
        self.action_logs = ["Game started. It is Player 1's turn!"]
        self._record_telemetry()

    def _record_telemetry(self):
        t = self.game.current_turn
        board = self.game.board
        if not self.match_telemetry["turns"] or self.match_telemetry["turns"][-1] != t:
            self.match_telemetry["turns"].append(t)
            for p in self.game.players:
                pid = p.player_id
                self.match_telemetry["cash"][pid].append(p.cash)
                self.match_telemetry["net_worth"][pid].append(p.net_worth(board))
                p_props = len([tile for tile in board.tiles if tile.owner == pid])
                self.match_telemetry["props"][pid].append(p_props)

    def setup_game(self, config: dict) -> dict[str, Any]:
        cfg_seed = config.get("seed")
        if cfg_seed is not None and cfg_seed > 0 and cfg_seed != 1234:
            self.seed = cfg_seed
        else:
            self.seed = random.randint(1, 99999999)
        max_turns = config.get("max_turns", 350)
        self.bot_chatter_rate = config.get("chatter_rate", getattr(self, "bot_chatter_rate", "normal"))
        self.recent_trades = []
        self.game = MonopolyGame(num_players=4, seed=self.seed, max_turns=max_turns, enable_logging=True)
        self.game.interactive_human = True
        self.match_stats = {pid: {"cash_spent": 0, "properties_bought": 0, "houses_built": 0, "rent_paid": 0, "taxes_paid": 0, "mortgages_taken": 0, "trades_completed": 0} for pid in range(4)}
        self.match_telemetry = {
            "turns": [],
            "cash": {0: [], 1: [], 2: [], 3: []},
            "net_worth": {0: [], 1: [], 2: [], 3: []},
            "props": {0: [], 1: [], 2: [], 3: []},
        }

        start_cash = config.get("starting_cash", 1500)
        for p in self.game.players:
            p.cash = start_cash

        player_cfgs = config.get("players", [])
        assigned_skins = set()
        for i in range(4):
            cfg = player_cfgs[i] if i < len(player_cfgs) else {}
            p_type = cfg.get("type", "human" if i == 0 else "rl" if i == 1 else "grandmaster" if i == 2 else "markov")
            default_name = "Player 1" if i == 0 else "Politically Perfect Organism" if p_type == "rl" else "Cash Elo" if p_type == "grandmaster" else "Moneybags Markov" if p_type == "markov" else f"Player {i}"
            p_name = cfg.get("name") or default_name
            p_skin = cfg.get("skin", i)
            if not isinstance(p_skin, int) or p_skin in assigned_skins or p_skin < 0 or p_skin > 7:
                for cand in range(8):
                    if cand not in assigned_skins:
                        p_skin = cand
                        break
            assigned_skins.add(p_skin)
            default_pers = "Deep Policy Gradient" if p_type == "rl" else "Chess Grandmaster" if p_type == "grandmaster" else "Stochastic Fatalist" if p_type == "markov" else "Organic Lifeform"
            p_personality = cfg.get("personality") or default_pers

            self.game.players[i].name = p_name
            self.player_skins[i] = p_skin
            self.player_types[i] = p_type
            self.player_personalities[i] = p_personality

            if p_type == "human":
                self.game.players[i].agent = None
            elif p_type == "rl":
                try:
                    self.game.players[i].agent = RLAgent(model_path=self.model_path, device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") != "" else "cpu")
                except Exception:
                    self.game.players[i].agent = RLAgent(model_path=self.model_path, device="cpu")
            elif p_type == "grandmaster":
                self.game.players[i].agent = TournamentGrandmasterAgent()
            elif p_type == "markov":
                self.game.players[i].agent = MarkovROIAgent()

        self.has_rolled = False
        self.can_roll_again = False
        self.last_dice = [0, 0]
        self.consecutive_doubles = 0
        self.active_counter_offer = None
        self.incoming_trade_proposal = None
        self.pending_auction = None
        self.last_human_tax_payment = None
        self.latest_speech = None
        self.game.auction_handler = self._handle_game_auction

        randomize_order = config.get("randomize_turn_order", True)
        if randomize_order:
            player_ids = list(range(4))
            initial_rolls = []
            rolls = {}
            roll_details = {}
            tie_logs = []
            tie_breakers_data = []

            for pid in player_ids:
                d1, d2 = random.randint(1, 6), random.randint(1, 6)
                rolls[pid] = d1 + d2
                roll_details[pid] = f"{d1}+{d2}={d1+d2}"
                initial_rolls.append({
                    "player_id": pid,
                    "player_name": self.game.players[pid].name,
                    "dice": [d1, d2],
                    "total": d1 + d2,
                })

            # Resolve any ties iteratively by re-rolling for tied players
            from collections import defaultdict
            by_val = defaultdict(list)
            for pid in player_ids:
                by_val[rolls[pid]].append(pid)

            sorted_order = []
            for val in sorted(by_val.keys(), reverse=True):
                group = by_val[val]
                if len(group) == 1:
                    sorted_order.append(group[0])
                else:
                    names = [self.game.players[p].name for p in group]
                    sub_rolls = {}
                    sub_dice = {}
                    while len(set(sub_rolls.values())) < len(group):
                        sub_rolls = {}
                        sub_dice = {}
                        for p in group:
                            td1, td2 = random.randint(1, 6), random.randint(1, 6)
                            sub_dice[p] = [td1, td2]
                            sub_rolls[p] = td1 + td2
                    reroll_strs = [f"{self.game.players[p].name}: {sub_rolls[p]}" for p in group]
                    tie_logs.append(f"Opening Roll Tie-breaker ({', '.join(names)}): {', '.join(reroll_strs)}.")
                    tie_breakers_data.append({
                        "player_names": names,
                        "rolls": [
                            {
                                "player_id": p,
                                "player_name": self.game.players[p].name,
                                "dice": sub_dice[p],
                                "total": sub_rolls[p],
                            }
                            for p in group
                        ]
                    })
                    sub_sorted = sorted(group, key=lambda p: sub_rolls[p], reverse=True)
                    sorted_order.extend(sub_sorted)

            self.turn_order = sorted_order
            self.game.turn_order = sorted_order
            self.game.turn_order_pos = 0
            starter_id = sorted_order[0]
            self.game.current_player_idx = starter_id

            opening_roll_lines = [f"{self.game.players[pid].name} ({roll_details[pid]})" for pid in sorted_order]
            starter_name = self.game.players[starter_id].name
            order_names = [f"#{idx+1} {self.game.players[pid].name}" for idx, pid in enumerate(sorted_order)]

            self.action_logs = [
                f"Opening High Roll: {', '.join(opening_roll_lines)}.",
            ]
            if tie_logs:
                self.action_logs.extend(tie_logs)
            self.action_logs.append(
                f"Turn Order: {' -> '.join(order_names)}. {starter_name} won high roll and moves first!"
            )

            if starter_id == 0:
                self.pending_decision = "ROLL"
            else:
                self.pending_decision = "AI_WAITING"

            self.opening_roll_ceremony = {
                "rolls": initial_rolls,
                "tie_breakers": tie_breakers_data,
                "turn_order": sorted_order,
                "starter_id": starter_id,
                "starter_name": starter_name,
            }
        else:
            self.turn_order = [0, 1, 2, 3]
            self.game.turn_order = [0, 1, 2, 3]
            self.game.turn_order_pos = 0
            self.game.current_player_idx = 0
            self.pending_decision = "ROLL"
            self.action_logs = ["Tournament arena initialized! Player 1 takes the opening turn."]
            self.opening_roll_ceremony = None

        self._record_telemetry()
        return self.get_state()

    def _handle_game_auction(self, tile) -> tuple[Optional[int], int]:
        p0 = self.game.players[0]
        if p0.is_bankrupt:
            return conduct_auction(tile, self.game.players, self.game.board)
        self.pending_auction = {
            "tile_index": tile.index,
            "tile_name": tile.name,
            "price": tile.price,
            "color_group": tile.color_group.name if tile.color_group else None,
            "mortgage_value": tile.mortgage_value,
        }
        self.game.log(f"AUCTION TRIGGERED for {tile.name} (Value: ${tile.price})! Bidding open.")
        return None, 0

    def submit_auction_bid(self, human_bid: int) -> dict[str, Any]:
        if not self.pending_auction:
            raise HTTPException(status_code=400, detail="No active auction pending.")
        pa = self.pending_auction
        tile = self.game.board.tiles[pa["tile_index"]]
        p0 = self.game.players[0]
        bid = max(0, min(human_bid, p0.cash)) if not p0.is_bankrupt else 0
        custom_vals = {0: bid}
        winner_id, winning_bid = conduct_auction(tile, self.game.players, self.game.board, custom_valuations=custom_vals)
        self.pending_auction = None
        if winner_id is not None:
            winner_name = self.game.players[winner_id].name
            self.log(f"AUCTION CONCLUDED: {winner_name} won {tile.name} for ${winning_bid}!")
            if winner_id == 0:
                self.log(f"You acquired {tile.name}! (Remaining Cash: ${p0.cash})")
        else:
            self.log(f"AUCTION CONCLUDED: No bids placed for {tile.name}.")
        return self.get_state()

    def negotiate_trade(self, message: str) -> dict[str, Any]:
        if not self.incoming_trade_proposal:
            raise HTTPException(status_code=400, detail="No active trade proposal to negotiate.")
        tp = self.incoming_trade_proposal
        bot_id = tp["bot_id"]
        bot_name = tp["bot_name"]
        curr_cash = tp.get("bot_gives_cash", 0)
        max_cash = tp.get("max_cash_willing", curr_cash + 100)

        nums = re.findall(r'\$?(\d+)', message)
        requested_cash = int(nums[0]) if nums else None

        if requested_cash is not None:
            if requested_cash <= curr_cash:
                tp["bot_gives_cash"] = requested_cash
                tp["offer_cash"] = requested_cash
                reply = f"Agreed! I will pay ${requested_cash} as requested. Accept the trade terms to finalize."
                can_raise = False
            elif requested_cash <= max_cash:
                tp["bot_gives_cash"] = requested_cash
                tp["offer_cash"] = requested_cash
                reply = f"Fair terms. I will meet your terms at ${requested_cash}. Do we have a deal?"
                can_raise = False
            else:
                if curr_cash < max_cash:
                    tp["bot_gives_cash"] = max_cash
                    tp["offer_cash"] = max_cash
                    reply = f"${requested_cash} exceeds my valuation ceiling, but I can stretch to ${max_cash}. That is my absolute highest offer."
                    can_raise = False
                else:
                    reply = f"No deal at ${requested_cash}. ${curr_cash} is my absolute limit."
                    can_raise = False
        else:
            can_raise = curr_cash < max_cash
            if can_raise:
                step_raise = min(max_cash - curr_cash, 35)
                tp["bot_gives_cash"] += step_raise
                tp["offer_cash"] = tp["bot_gives_cash"]
                tp["negotiation_step"] = tp.get("negotiation_step", 1) + 1
                reply = f"Fair point. I will increase my cash offer to ${tp['bot_gives_cash']}. Do we have a deal?"
            else:
                reply = f"${curr_cash} is my absolute limit. I cannot justify offering more for this property."

        self.log(f"TRADE NEGOTIATION: {bot_name}: '{reply}'")
        return {
            "response": reply,
            "new_cash": tp["bot_gives_cash"],
            "max_reached": not can_raise,
            "state": self.get_state(),
        }

    def log(self, msg: str):
        self.action_logs.append(msg)
        if len(self.action_logs) > 60:
            self.action_logs.pop(0)

    def get_state(self) -> dict[str, Any]:
        p0 = self.game.players[0]
        board = self.game.board

        # Buildable groups for Human
        buildable = []
        sellable = []
        for g in BUILDABLE_GROUPS:
            if board.owns_full_group(0, g):
                indices = COLOR_GROUP_TILES[g]
                is_mortgaged = any(board.tiles[i].is_mortgaged for i in indices)
                if not is_mortgaged:
                    min_dev = min(board.tiles[i].num_houses + (5 if board.tiles[i].num_hotels else 0) for i in indices)
                    cost = board.tiles[indices[0]].house_cost
                    tier_cost = cost * len(indices)
                    if min_dev < 5 and p0.cash >= cost:
                        buildable.append({
                            "group": g.name,
                            "group_id": g.value,
                            "min_dev": min_dev,
                            "house_cost": cost,
                            "tier_cost": tier_cost,
                            "indices": indices,
                            "names": [board.tiles[i].name for i in indices],
                        })
                max_dev = max(board.tiles[i].num_houses + (5 if board.tiles[i].num_hotels else 0) for i in indices)
                if max_dev > 0:
                    sellable.append({
                        "group": g.name,
                        "group_id": g.value,
                        "max_dev": max_dev,
                        "indices": indices,
                    })

        # Mortgagable / Unmortgagable for Human
        mortgagable = []
        unmortgagable = []
        for idx in ALL_PROPERTY_INDICES:
            tile = board.tiles[idx]
            if tile.owner == 0:
                if board.can_mortgage(idx, 0):
                    mortgagable.append({
                        "index": idx,
                        "name": tile.name,
                        "mortgage_value": tile.mortgage_value,
                    })
                if board.can_unmortgage(idx, 0):
                    cost = int(tile.mortgage_value * 1.1)
                    unmortgagable.append({
                        "index": idx,
                        "name": tile.name,
                        "unmortgage_cost": cost,
                        "can_afford": p0.cash >= cost,
                    })

        # Landed tile details if BUY_OR_AUCTION
        landed_tile_info = None
        if self.pending_decision == "BUY_OR_AUCTION":
            t = board.tiles[p0.position]
            landed_tile_info = {
                "index": t.index,
                "name": t.name,
                "price": t.price,
                "mortgage_value": t.mortgage_value,
                "house_cost": t.house_cost,
                "base_rent": t.base_rent,
                "rents": [t.base_rent, t.rent_1, t.rent_2, t.rent_3, t.rent_4, t.rent_hotel],
                "color_group": t.color_group.name if t.color_group else None,
                "can_afford": p0.cash >= t.price,
            }

        # Players state
        players_data = []
        for p in self.game.players:
            props = [t.index for t in board.tiles if t.owner == p.player_id]
            players_data.append({
                "id": p.player_id,
                "name": p.name,
                "cash": p.cash,
                "position": p.position,
                "in_jail": p.in_jail,
                "jail_turns": p.jail_turns,
                "jail_cards": p.get_out_of_jail_cards,
                "is_bankrupt": p.is_bankrupt,
                "net_worth": p.net_worth(board),
                "properties": props,
                "skin": self.player_skins.get(p.player_id, p.player_id),
                "type": self.player_types.get(p.player_id, "human" if p.player_id == 0 else "bot"),
                "personality": self.player_personalities.get(p.player_id, ""),
            })

        # Tiles state
        tiles_data = []
        for t in board.tiles:
            tiles_data.append({
                "index": t.index,
                "name": t.name,
                "owner": t.owner,
                "is_mortgaged": t.is_mortgaged,
                "num_houses": t.num_houses,
                "num_hotels": t.num_hotels,
                "price": t.price,
                "mortgage_value": t.mortgage_value,
                "house_cost": t.house_cost,
                "base_rent": t.base_rent,
                "rents": [t.base_rent, t.rent_1, t.rent_2, t.rent_3, t.rent_4, t.rent_hotel],
                "tile_type": t.tile_type.name,
                "color_group": t.color_group.name if t.color_group else None,
            })

        is_spectator = bool(p0.is_bankrupt)

        return {
            "turn": self.game.current_turn,
            "max_turns": self.game.max_turns,
            "active_player": self.game.current_player_idx,
            "turn_order": getattr(self, "turn_order", [0, 1, 2, 3]),
            "opening_roll_ceremony": getattr(self, "opening_roll_ceremony", None),
            "game_over": self.game.game_over,
            "winner_id": self.game.winner_id,
            "winner_name": self.game.players[self.game.winner_id].name if self.game.winner_id is not None else None,
            "has_rolled": self.has_rolled,
            "can_roll_again": self.can_roll_again,
            "spectator_mode": is_spectator,
            "pending_decision": self.pending_decision if not is_spectator else "SPECTATING",
            "last_dice": self.last_dice,
            "player_skins": self.player_skins,
            "available_houses": board.available_houses,
            "available_hotels": board.available_hotels,
            "players": players_data,
            "tiles": tiles_data,
            "landed_tile": landed_tile_info,
            "buildable_groups": buildable,
            "sellable_groups": sellable,
            "mortgagable_props": mortgagable,
            "unmortgagable_props": unmortgagable,
            "chance_cards_remaining": len(board.chance_deck.cards),
            "chest_cards_remaining": len(board.community_chest_deck.cards),
            "chance_deck_cards": [c.name for c in board.chance_deck.cards],
            "chance_discard_cards": [c.name for c in board.chance_deck.discard_pile],
            "chest_deck_cards": [c.name for c in board.community_chest_deck.cards],
            "chest_discard_cards": [c.name for c in board.community_chest_deck.discard_pile],
            "incoming_trade_proposal": self.incoming_trade_proposal,
            "match_telemetry": self.match_telemetry,
            "logs": self.action_logs[-45:],
            "last_card_drawn": getattr(self.game, "last_card_drawn", None),
            "last_jail_event": getattr(self.game, "last_jail_event", None),
            "latest_speech": self.latest_speech,
            "active_counter_offer": self.active_counter_offer,
            "diplomatic_ledger": self.diplomatic_ledger,
            "pending_auction": self.pending_auction,
            "last_tax_event": self.last_human_tax_payment,
            "bot_chatter_rate": getattr(self, "bot_chatter_rate", "normal"),
        }

    def roll_dice(self) -> dict[str, Any]:
        if self.game.current_player_idx != 0:
            raise HTTPException(status_code=400, detail="It is not your turn!")
        p0 = self.game.players[0]
        if p0.is_bankrupt:
            raise HTTPException(status_code=400, detail="You are bankrupt!")
        if self.has_rolled and not self.can_roll_again:
            raise HTTPException(status_code=400, detail="You already rolled this turn!")

        self.game.last_card_drawn = None
        self.game.last_jail_event = None
        board = self.game.board
        d1, d2, is_double = self.game.roll_dice()
        dice_sum = d1 + d2
        self.last_dice = [d1, d2]

        if p0.in_jail:
            self.can_roll_again = False
            if is_double:
                p0.in_jail = False
                p0.jail_turns = 0
                self.log(f"You rolled doubles ({d1}, {d2}) and broke out of Jail!")
                p0.position = (p0.position + dice_sum) % 40
                self._handle_landing(p0, dice_sum)
                self.has_rolled = True
            elif p0.jail_turns >= 2:
                if self.game.handle_payment(p0, None, JAIL_FINE):
                    p0.in_jail = False
                    p0.jail_turns = 0
                    self.log(f"3rd turn in Jail: Paid ${JAIL_FINE} fine and rolled ({d1}, {d2}).")
                    p0.position = (p0.position + dice_sum) % 40
                    self._handle_landing(p0, dice_sum)
                else:
                    self.log(f"Bankrupt trying to pay Jail fine!")
                    self.pending_decision = "END_TURN"
                self.has_rolled = True
            else:
                p0.jail_turns += 1
                self.log(f"You rolled ({d1}, {d2}) - no doubles. Still in Jail (turn {p0.jail_turns}/3).")
                self.has_rolled = True
                self.pending_decision = "END_TURN"
                return self.get_state()
        else:
            if is_double:
                self.consecutive_doubles += 1
                if self.consecutive_doubles >= 3:
                    self.game.send_to_jail(p0)
                    self.log(f"Rolled doubles 3 times in a row! Sent directly to Jail!")
                    self.has_rolled = True
                    self.can_roll_again = False
                    self.pending_decision = "END_TURN"
                    return self.get_state()
                else:
                    self.log(f"You rolled doubles ({d1}, {d2}) = {dice_sum}! You get to roll again.")
                    self.can_roll_again = True
                    self.has_rolled = False
            else:
                self.consecutive_doubles = 0
                self.can_roll_again = False
                self.has_rolled = True
                self.log(f"You rolled ({d1}, {d2}) = {dice_sum}.")

            new_pos = (p0.position + dice_sum) % 40
            if new_pos < p0.position:
                p0.cash += 200
                self.log(f"You passed GO! Collected $200 salary.")
            p0.position = new_pos
            self._handle_landing(p0, dice_sum)

        return self.get_state()

    def _handle_landing(self, player, dice_sum):
        tile = self.game.board.tiles[player.position]
        self.log(f"You landed on {tile.name} ({tile.index}).")
        self.last_human_tax_payment = None

        if tile.is_purchasable and tile.owner is None:
            self.pending_decision = "BUY_OR_AUCTION"
        else:
            start_log_idx = len(self.game.log_messages)
            self.game.resolve_tile_landing(player, dice_sum)
            new_logs = self.game.log_messages[start_log_idx:]
            for l in new_logs:
                self.log(l)

            # Check if card moved player to an unowned purchasable property
            dest_tile = self.game.board.tiles[player.position]
            if dest_tile.is_purchasable and dest_tile.owner is None:
                self.pending_decision = "BUY_OR_AUCTION"
                return

            if tile.tile_type == TileType.TAX:
                tax_amt = 200 if tile.index == 4 else 100
                self.match_stats[0]["taxes_paid"] += tax_amt
                self.match_stats[0]["cash_spent"] += tax_amt
                self.last_human_tax_payment = {
                    "tile_index": tile.index,
                    "tile_name": tile.name,
                    "amount": tax_amt,
                    "remaining_cash": player.cash,
                }
            if self.can_roll_again and not player.in_jail and not player.is_bankrupt:
                self.pending_decision = "ROLL"
            else:
                self.pending_decision = "END_TURN"

    def buy_property(self) -> dict[str, Any]:
        if self.game.current_player_idx != 0:
            raise HTTPException(status_code=400, detail="It is not your turn!")
        p0 = self.game.players[0]
        tile = self.game.board.tiles[p0.position]
        if self.pending_decision != "BUY_OR_AUCTION":
            raise HTTPException(status_code=400, detail="No pending property to buy.")
        if p0.cash < tile.price:
            raise HTTPException(status_code=400, detail="Insufficient cash to buy.")

        p0.cash -= tile.price
        tile.owner = 0
        self.match_stats[0]["cash_spent"] += tile.price
        self.match_stats[0]["properties_bought"] += 1
        self.log(f"You bought {tile.name} for ${tile.price}! (Cash: ${p0.cash})")

        if self.can_roll_again and not p0.in_jail:
            self.pending_decision = "ROLL"
            self.has_rolled = False
        else:
            self.pending_decision = "END_TURN"
        return self.get_state()

    def decline_auction(self, human_bid: int = 0) -> dict[str, Any]:
        if self.game.current_player_idx != 0:
            raise HTTPException(status_code=400, detail="It is not your turn!")
        p0 = self.game.players[0]
        tile = self.game.board.tiles[p0.position]
        if self.pending_decision != "BUY_OR_AUCTION":
            raise HTTPException(status_code=400, detail="No pending property to auction.")

        self.log(f"You passed {tile.name} to auction (Your max bid: ${human_bid}).")

        # Custom valuations: if human_bid is 0, human NEVER bids in this auction
        custom_vals = {0: min(human_bid, p0.cash) if human_bid > 0 else 0}
        winner_id, winning_bid = conduct_auction(tile, self.game.players, self.game.board, custom_valuations=custom_vals)

        if winner_id is not None:
            winner_name = self.game.players[winner_id].name
            self.log(f"AUCTION: {winner_name} won {tile.name} for ${winning_bid}!")
            self.match_stats[winner_id]["cash_spent"] += winning_bid
            self.match_stats[winner_id]["properties_bought"] += 1
            if winner_id != 0:
                self.latest_speech = {
                    "speaker_id": winner_id,
                    "speaker_name": winner_name,
                    "text": self.llm_strategist.generate_trash_talk(winner_id, winner_name, "BUY", target_name=p0.name, tile_name=tile.name, amount=winning_bid),
                    "event": "AUCTION_WIN"
                }
        else:
            self.log(f"AUCTION: No player bid on {tile.name}.")

        if self.can_roll_again and not p0.in_jail:
            self.pending_decision = "ROLL"
            self.has_rolled = False
        else:
            self.pending_decision = "END_TURN"
        return self.get_state()

    def build_tier(self, group_name: str) -> dict[str, Any]:
        p0 = self.game.players[0]
        group = getattr(ColorGroup, group_name, None)
        if group is None or not self.game.board.owns_full_group(0, group):
            raise HTTPException(status_code=400, detail="You do not own this color group.")

        indices = COLOR_GROUP_TILES[group]
        cost = self.game.board.tiles[indices[0]].house_cost
        tier_cost = cost * len(indices)

        if p0.cash < tier_cost:
            raise HTTPException(status_code=400, detail="Insufficient cash to build tier.")

        p0.cash -= tier_cost
        self.match_stats[0]["cash_spent"] += tier_cost
        self.match_stats[0]["houses_built"] += len(indices)
        for idx in indices:
            t = self.game.board.tiles[idx]
            if t.num_houses == 4:
                t.num_houses = 0
                t.num_hotels = 1
                self.game.board.available_houses += 4
                self.game.board.available_hotels -= 1
            else:
                t.num_houses += 1
                self.game.board.available_houses -= 1

        self.log(f"You developed {group_name} properties for -${tier_cost}! (Cash: ${p0.cash})")
        return self.get_state()

    def mortgage_property(self, prop_idx: int) -> dict[str, Any]:
        p0 = self.game.players[0]
        tile = self.game.board.tiles[prop_idx]
        if not self.game.board.can_mortgage(prop_idx, 0):
            raise HTTPException(status_code=400, detail="Cannot mortgage this property.")
        p0.cash += tile.mortgage_value
        tile.is_mortgaged = True
        self.match_stats[0]["mortgages_taken"] += 1
        self.log(f"You mortgaged {tile.name} for +${tile.mortgage_value}. (Cash: ${p0.cash})")
        return self.get_state()

    def unmortgage_property(self, prop_idx: int) -> dict[str, Any]:
        p0 = self.game.players[0]
        tile = self.game.board.tiles[prop_idx]
        cost = int(tile.mortgage_value * 1.1)
        if not self.game.board.can_unmortgage(prop_idx, 0) or p0.cash < cost:
            raise HTTPException(status_code=400, detail="Cannot unmortgage this property.")
        p0.cash -= cost
        tile.is_mortgaged = False
        self.log(f"You unmortgaged {tile.name} for -${cost}. (Cash: ${p0.cash})")
        return self.get_state()

    def jail_action(self, action: str) -> dict[str, Any]:
        p0 = self.game.players[0]
        if not p0.in_jail:
            raise HTTPException(status_code=400, detail="You are not in Jail.")
        if action == "PAY":
            if p0.cash < JAIL_FINE:
                raise HTTPException(status_code=400, detail="Insufficient cash for jail fine.")
            p0.cash -= JAIL_FINE
            p0.in_jail = False
            p0.jail_turns = 0
            self.has_rolled = False
            self.can_roll_again = False
            self.pending_decision = "ROLL"
            self.log(f"Paid ${JAIL_FINE} fine to get out of Jail. Rolling now...")
            return self.roll_dice()
        elif action == "CARD":
            if p0.get_out_of_jail_cards <= 0:
                raise HTTPException(status_code=400, detail="You have no Get Out of Jail Free card.")
            p0.get_out_of_jail_cards -= 1
            p0.in_jail = False
            p0.jail_turns = 0
            self.has_rolled = False
            self.can_roll_again = False
            self.pending_decision = "ROLL"
            self.log(f"Used Get Out of Jail Free card! Rolling now...")
            return self.roll_dice()
        elif action == "ROLL":
            return self.roll_dice()
        return self.get_state()

    def end_turn(self) -> dict[str, Any]:
        p0 = self.game.players[0]
        if self.can_roll_again and not p0.in_jail and not p0.is_bankrupt:
            self.log("You rolled doubles! You must roll again before ending your turn.")
            return self.get_state()

        self.consecutive_doubles = 0
        self.can_roll_again = False
        self.has_rolled = False
        self.game._advance_turn()

        # If human went bankrupt, keep advancing until game over or active bot
        while self.game.current_player_idx != 0 and not self.game.game_over:
            opp = self.game.players[self.game.current_player_idx]
            if not opp.is_bankrupt:
                start_log_idx = len(self.game.log_messages)
                self.game.execute_turn()
                new_logs = self.game.log_messages[start_log_idx:]
                for l in new_logs:
                    self.log(l)
            else:
                self.game._advance_turn()
            self.game._check_game_over()

        if self.game.game_over:
            winner = self.game.players[self.game.winner_id].name if self.game.winner_id is not None else "Draw"
            if not any("GAME OVER!" in l for l in self.action_logs[-3:]):
                self.log(f"GAME OVER! Winner: {winner}")
            self.pending_decision = "GAME_OVER"
        elif p0.is_bankrupt:
            self.pending_decision = "SPECTATING"
        else:
            self.pending_decision = "JAIL" if self.game.players[0].in_jail else "ROLL"
            self.log(f"--- Turn {self.game.current_turn}: It is your turn! ---")

        return self.get_state()

    def start_ai_turns(self) -> dict[str, Any]:
        p0 = self.game.players[0]
        if self.incoming_trade_proposal is not None or self.pending_auction is not None:
            return {
                "done": True,
                "active_player": self.game.current_player_idx,
                "can_reroll": False,
                "state": self.get_state(),
            }

        if self.can_roll_again and not p0.in_jail and not p0.is_bankrupt:
            self.log("You rolled doubles! You must roll again.")
            return {"done": True, "active_player": 0, "can_reroll": True, "state": self.get_state()}

        if self.game.current_player_idx == 0:
            self.consecutive_doubles = 0
            self.can_roll_again = False
            self.has_rolled = False
            self.game._advance_turn()

        return {
            "done": self.game.game_over or (self.game.current_player_idx == 0 and not p0.is_bankrupt),
            "active_player": self.game.current_player_idx,
            "can_reroll": False,
            "state": self.get_state(),
        }

    def step_ai_turn(self) -> dict[str, Any]:
        p0 = self.game.players[0]
        if self.game.game_over:
            return {"done": True, "active_player": self.game.current_player_idx, "state": self.get_state()}

        if self.incoming_trade_proposal is not None:
            return {
                "done": True,
                "trade_proposed": True,
                "auction_pending": False,
                "active_player": self.game.current_player_idx,
                "state": self.get_state(),
            }

        if self.pending_auction is not None:
            return {
                "done": True,
                "trade_proposed": False,
                "auction_pending": True,
                "pending_auction": self.pending_auction,
                "active_player": self.game.current_player_idx,
                "state": self.get_state(),
            }

        # Advance past any bankrupt players until we find an active player or reach active human or game over
        loop_guard = 0
        while not self.game.game_over and loop_guard < 12:
            loop_guard += 1
            if self.game.current_player_idx == 0 and not p0.is_bankrupt:
                break
            curr_p = self.game.players[self.game.current_player_idx]
            if not curr_p.is_bankrupt and self.game.current_player_idx != 0:
                break
            self.game._advance_turn()
            self.game._check_game_over()

        if self.game.game_over or (self.game.current_player_idx == 0 and not p0.is_bankrupt):
            return {"done": True, "active_player": self.game.current_player_idx, "state": self.get_state()}

        ai_player = self.game.players[self.game.current_player_idx]
        from_pos = ai_player.position
        from_cash = ai_player.cash
        start_log_idx = len(self.game.log_messages)

        self.game.execute_turn()
        self.game._check_game_over()
        new_logs = self.game.log_messages[start_log_idx:]
        for l in new_logs:
            self.log(l)

        # Telemetry updates for AI actions
        for l in new_logs:
            if "bought " in l and " for $" in l:
                self.match_stats[ai_player.player_id]["properties_bought"] += 1
                try:
                    price_val = int(l.split(" for $")[-1].replace(".", "").strip())
                    self.match_stats[ai_player.player_id]["cash_spent"] += price_val
                except Exception:
                    pass
            elif "built house #" in l or "built a HOTEL" in l:
                self.match_stats[ai_player.player_id]["houses_built"] += 1
            elif "paid $" in l and "tax" in l.lower():
                try:
                    tax_val = int(l.split("paid $")[-1].split(" ")[0].replace(".", "").strip())
                    self.match_stats[ai_player.player_id]["taxes_paid"] += tax_val
                    self.match_stats[ai_player.player_id]["cash_spent"] += tax_val
                except Exception:
                    pass
            elif "paid $" in l and "rent" in l.lower():
                try:
                    rent_val = int(l.split("paid $")[-1].split(" ")[0].replace(".", "").strip())
                    self.match_stats[ai_player.player_id]["rent_paid"] += rent_val
                    self.match_stats[ai_player.player_id]["cash_spent"] += rent_val
                except Exception:
                    pass

        to_pos = ai_player.position
        to_cash = ai_player.cash
        d1, d2 = self.game.last_dice

        # In-character bot speech triggers with controlled event probability and bot-to-bot dialogue
        self.latest_speech = None
        speech_events: list[dict] = []
        should_speak = False
        speech_text = None
        target_player_id = None
        target_player_name = None
        speech_event = None

        # Check for rent transactions in logs
        rent_logs = [l for l in new_logs if "paid $" in l and "rent" in l]
        if to_cash < from_cash and rent_logs:
            delta = from_cash - to_cash
            recipient_name = None
            recipient_id = None
            for p in self.game.players:
                if p.player_id != ai_player.player_id and p.name in rent_logs[0]:
                    recipient_name = p.name
                    recipient_id = p.player_id
                    break

            if delta >= 40 or random.random() < 0.35:
                should_speak = True
                speech_event = "RENT_PAY"
                target_player_id = recipient_id
                target_player_name = recipient_name or "Table"
                if recipient_name:
                    speech_text = self.llm_strategist.generate_bot_banter(
                        ai_player.player_id, ai_player.name, recipient_id, recipient_name, "RENT_PAY", amount=delta
                    )
                else:
                    speech_text = self.llm_strategist.generate_trash_talk(
                        ai_player.player_id, ai_player.name, "RENT_PAY", amount=delta
                    )
        elif to_cash > from_cash and any("collected $" in l and "rent" in l for l in new_logs):
            delta = to_cash - from_cash
            if delta >= 40 or random.random() < 0.35:
                should_speak = True
                speech_event = "RENT_COLLECT"
                speech_text = self.llm_strategist.generate_trash_talk(
                    ai_player.player_id, ai_player.name, "RENT_COLLECT", amount=delta
                )
        elif getattr(self.game, "last_jail_event", None):
            if random.random() < 0.70:
                should_speak = True
                speech_event = "JAIL"
                speech_text = self.llm_strategist.generate_trash_talk(
                    ai_player.player_id, ai_player.name, "JAIL"
                )
        elif any("is BANKRUPT" in l or "Bankrupt" in l for l in new_logs):
            should_speak = True
            speech_event = "BANKRUPTCY"
            speech_text = self.llm_strategist.generate_trash_talk(
                ai_player.player_id, ai_player.name, "BANKRUPTCY"
            )
        elif any("bought " in l for l in new_logs):
            if random.random() < 0.30:
                bought_log = [l for l in new_logs if "bought " in l][0]
                tile_n = bought_log.split("bought ")[-1].split(" for")[0]
                should_speak = True
                speech_event = "BUY"
                speech_text = self.llm_strategist.generate_trash_talk(
                    ai_player.player_id, ai_player.name, "BUY", tile_name=tile_n
                )

        # Spontaneous table commentary if no event-driven speech occurred
        if not should_speak and getattr(self, "bot_chatter_rate", "normal") != "silent":
            chatter_probs = {"high": 0.50, "normal": 0.25, "low": 0.08}
            c_rate = getattr(self, "bot_chatter_rate", "normal")
            if random.random() < chatter_probs.get(c_rate, 0.25):
                surviving = [p for p in self.game.players if not p.is_bankrupt]
                leader = max(surviving, key=lambda p: p.net_worth(self.game.board)) if surviving else None
                threat = max(surviving, key=lambda p: len([t for t in self.game.board.tiles if t.owner == p.player_id])) if surviving else None
                leader_n = leader.name if leader and leader.player_id != ai_player.player_id else None
                threat_n = threat.name if threat and threat.player_id != ai_player.player_id else None

                sp_text = self.llm_strategist.generate_spontaneous_chatter(
                    ai_player.player_id, ai_player.name, leader_name=leader_n, threat_name=threat_n
                )
                if sp_text:
                    should_speak = True
                    speech_event = "SPONTANEOUS"
                    speech_text = sp_text
                    self.log(f"CHAT: {ai_player.name}: \"{sp_text}\"")

        if should_speak and speech_text:
            self.latest_speech = {
                "speaker_id": ai_player.player_id,
                "speaker_name": ai_player.name,
                "target_id": target_player_id,
                "target_name": target_player_name or "Table",
                "text": speech_text,
                "event": speech_event,
                "channel": "table"
            }
            speech_events.append(self.latest_speech)

        def is_safe_trade_prop(prop, recipient_id, board):
            if not prop or not prop.color_group or prop.color_group not in COLOR_GROUP_TILES:
                return True
            g_tiles = COLOR_GROUP_TILES[prop.color_group]
            recipient_owns = sum(1 for idx in g_tiles if board.tiles[idx].owner == recipient_id)
            if recipient_owns >= (len(g_tiles) - 1):
                return False
            return True

        # Check for trade proposals ONLY if no auction is pending and no trade is already pending
        if self.pending_auction is None and self.incoming_trade_proposal is None and not ai_player.is_bankrupt and ai_player.cash >= 120:
            board = self.game.board
            curr_turn = self.game.current_turn
            recently_traded_ids = {prop_id for (turn, prop_id) in self.recent_trades if (curr_turn - turn) < 15}

            def try_bot_trade() -> bool:
                other_bots = [p for p in self.game.players if p.player_id != 0 and p.player_id != ai_player.player_id and not p.is_bankrupt]
                if not other_bots or random.random() > 0.40:
                    return False
                target_bot = random.choice(other_bots)
                bot_target_tile = None
                for g in BUILDABLE_GROUPS:
                    indices = COLOR_GROUP_TILES[g]
                    has_ai = any(board.tiles[idx].owner == ai_player.player_id for idx in indices)
                    has_tb = [board.tiles[idx] for idx in indices if board.tiles[idx].owner == target_bot.player_id and not board.tiles[idx].is_mortgaged and board.tiles[idx].index not in recently_traded_ids]
                    if has_ai and has_tb:
                        bot_target_tile = has_tb[0]
                        break

                if not bot_target_tile:
                    return False

                give_prop = None
                ai_props = [t for t in board.tiles if t.owner == ai_player.player_id and t.is_purchasable and not t.is_mortgaged]
                for ap in ai_props:
                    if ap.color_group and ap.color_group != bot_target_tile.color_group and not board.owns_full_group(ai_player.player_id, ap.color_group) and ap.index not in recently_traded_ids:
                        if is_safe_trade_prop(ap, target_bot.player_id, board):
                            give_prop = ap
                            break

                cash_offer = max(30, min(int(ai_player.cash * 0.25), bot_target_tile.price + 50))
                give_str = f"{give_prop.name} + ${cash_offer} cash" if give_prop else f"${cash_offer} cash"
                p_text = f"{target_bot.name}, I propose offering you {give_str} for {bot_target_tile.name}. Deal?"
                self.log(f"BOT DIPLOMACY: {ai_player.name} -> {target_bot.name}: \"{p_text}\"")

                offered_props = [{"index": give_prop.index, "name": give_prop.name, "price": give_prop.price}] if give_prop else []
                req_props = [{"index": bot_target_tile.index, "name": bot_target_tile.name, "price": bot_target_tile.price}]
                board_state = {
                    "player_props": {p.player_id: [t.index for t in board.tiles if t.owner == p.player_id] for p in self.game.players}
                }
                eval_res = self.llm_strategist.evaluate_trade(
                    bot_id=target_bot.player_id,
                    bot_name=target_bot.name,
                    proposer_id=ai_player.player_id,
                    proposer_name=ai_player.name,
                    offered_props=offered_props,
                    offered_cash=cash_offer,
                    requested_props=req_props,
                    requested_cash=0,
                    board_state=board_state,
                    diplomatic_history={},
                )
                if eval_res["accept"]:
                    if give_prop:
                        give_prop.owner = target_bot.player_id
                        self.recent_trades.append((curr_turn, give_prop.index))
                    bot_target_tile.owner = ai_player.player_id
                    self.recent_trades.append((curr_turn, bot_target_tile.index))
                    ai_player.cash -= cash_offer
                    target_bot.cash += cash_offer
                    self.match_stats[ai_player.player_id]["trades_completed"] += 1
                    self.match_stats[target_bot.player_id]["trades_completed"] += 1
                    self.match_stats[ai_player.player_id]["cash_spent"] += cash_offer
                    r_text = f"Deal accepted, {ai_player.name}! A mutually beneficial agreement."
                    self.log(f"BOT DIPLOMACY: {target_bot.name} -> {ai_player.name}: \"{r_text}\"")
                    self.log(f"BOT TRADE EXECUTED: {ai_player.name} traded {give_str} to {target_bot.name} for {bot_target_tile.name}!")
                    speech_events.append({
                        "speaker_id": ai_player.player_id,
                        "speaker_name": ai_player.name,
                        "target_id": target_bot.player_id,
                        "target_name": target_bot.name,
                        "text": p_text,
                        "reply_text": r_text,
                        "event": "BOT_TRADE",
                        "channel": "table"
                    })
                else:
                    r_text = f"No deal, {ai_player.name}. {bot_target_tile.name} is too valuable to relinquish."
                    self.log(f"BOT DIPLOMACY: {target_bot.name} -> {ai_player.name}: \"{r_text}\"")
                    self.log(f"BOT TRADE DECLINED: {target_bot.name} declined trade offer from {ai_player.name}.")
                    speech_events.append({
                        "speaker_id": ai_player.player_id,
                        "speaker_name": ai_player.name,
                        "target_id": target_bot.player_id,
                        "target_name": target_bot.name,
                        "text": p_text,
                        "reply_text": r_text,
                        "event": "BOT_TRADE_DECLINE",
                        "channel": "table"
                    })
                return True

            def try_human_trade() -> bool:
                if p0.is_bankrupt or random.random() > 0.40:
                    return False
                ai_props = [t for t in board.tiles if t.owner == ai_player.player_id and t.is_purchasable]
                target_p0_tile = None
                for g in BUILDABLE_GROUPS:
                    indices = COLOR_GROUP_TILES[g]
                    has_ai = any(board.tiles[idx].owner == ai_player.player_id for idx in indices)
                    has_p0 = [board.tiles[idx] for idx in indices if board.tiles[idx].owner == 0 and board.tiles[idx].index not in recently_traded_ids]
                    if has_ai and has_p0:
                        target_p0_tile = has_p0[0]
                        break

                if not target_p0_tile:
                    return False

                give_prop = None
                for ap in ai_props:
                    g = ap.color_group
                    if g and g != target_p0_tile.color_group and ap.index not in recently_traded_ids:
                        if not is_safe_trade_prop(ap, 0, board):
                            continue
                        ai_group_count = sum(1 for idx in COLOR_GROUP_TILES[g] if board.tiles[idx].owner == ai_player.player_id)
                        if ai_group_count == 1:
                            give_prop = ap
                            break
                if not give_prop:
                    for ap in ai_props:
                        g = ap.color_group
                        if g and g != target_p0_tile.color_group and not board.owns_full_group(ai_player.player_id, g) and ap.index not in recently_traded_ids:
                            if is_safe_trade_prop(ap, 0, board):
                                give_prop = ap
                                break

                lowball_cash = max(15, min(int(ai_player.cash * 0.15), int(target_p0_tile.price * 0.4)))
                max_cash = min(int(ai_player.cash * 0.40), target_p0_tile.price + 120)
                give_prop_ids = [give_prop.index] if give_prop else []
                give_prop_names = [give_prop.name] if give_prop else []
                offer_desc = f"{give_prop.name} + ${lowball_cash} cash" if give_prop else f"${lowball_cash} cash"
                p_msg = f"{ai_player.name}: Player 1, I need {target_p0_tile.name}. I propose offering you {offer_desc} for it. Deal?"

                self.incoming_trade_proposal = {
                    "bot_id": ai_player.player_id,
                    "bot_name": ai_player.name,
                    "bot_gives_props": give_prop_ids,
                    "bot_gives_props_names": give_prop_names,
                    "bot_gives_cash": lowball_cash,
                    "max_cash_willing": max_cash,
                    "negotiation_step": 1,
                    "bot_wants_props": [target_p0_tile.index],
                    "bot_wants_props_names": [target_p0_tile.name],
                    "bot_wants_cash": 0,
                    "proposal_message": p_msg,
                    "offer_props": give_prop_ids,
                    "offer_cash": lowball_cash,
                    "request_props": [target_p0_tile.index],
                    "request_cash": 0,
                    "speech": p_msg,
                }
                self.log(f"TRADE INITIATION: {p_msg}")
                return True

            # 50% chance to consider bot trade first, otherwise human trade first
            if random.random() < 0.50:
                if not try_bot_trade():
                    try_human_trade()
            else:
                if not try_human_trade():
                    try_bot_trade()

        self._record_telemetry()

        if self.game.game_over:
            winner = self.game.players[self.game.winner_id].name if self.game.winner_id is not None else "Draw"
            if not any("GAME OVER!" in l for l in self.action_logs[-3:]):
                self.log(f"GAME OVER! Winner: {winner}")
            self.pending_decision = "GAME_OVER"
        elif self.game.current_player_idx == 0 and not p0.is_bankrupt:
            self.pending_decision = "JAIL" if self.game.players[0].in_jail else "ROLL"
            self.has_rolled = False
            self.consecutive_doubles = 0
            self.can_roll_again = False
            self.log(f"--- Turn {self.game.current_turn}: It is Player 1's turn! ---")
        elif p0.is_bankrupt:
            self.pending_decision = "SPECTATING"

        is_done = self.game.game_over or (self.game.current_player_idx == 0 and not p0.is_bankrupt) or (self.incoming_trade_proposal is not None) or (self.pending_auction is not None)

        return {
            "done": is_done,
            "trade_proposed": bool(self.incoming_trade_proposal),
            "auction_pending": bool(self.pending_auction),
            "pending_auction": self.pending_auction,
            "spectator_mode": p0.is_bankrupt,
            "ai_player_id": ai_player.player_id,
            "ai_player_name": ai_player.name,
            "dice": [d1, d2],
            "from_pos": from_pos,
            "to_pos": to_pos,
            "cash_delta": to_cash - from_cash,
            "step_logs": new_logs,
            "last_card_drawn": getattr(self.game, "last_card_drawn", None),
            "last_jail_event": getattr(self.game, "last_jail_event", None),
            "latest_speech": self.latest_speech,
            "speech_events": speech_events,
            "active_player": self.game.current_player_idx,
            "state": self.get_state(),
        }

    def handle_trade(self, target_id: int, offer_props: list[int], offer_cash: int, request_props: list[int], request_cash: int) -> dict[str, Any]:
        p0 = self.game.players[0]
        target = self.game.players[target_id]
        board = self.game.board

        if p0.cash < offer_cash:
            raise HTTPException(status_code=400, detail="You do not have enough cash for this offer.")
        if target.cash < request_cash:
            raise HTTPException(status_code=400, detail=f"{target.name} does not have enough cash for this request.")

        for idx in offer_props:
            if board.tiles[idx].owner != 0:
                raise HTTPException(status_code=400, detail=f"You do not own {board.tiles[idx].name}.")
        for idx in request_props:
            if board.tiles[idx].owner != target_id:
                raise HTTPException(status_code=400, detail=f"{target.name} does not own {board.tiles[idx].name}.")

        offered_prop_dicts = [{"index": idx, "name": board.tiles[idx].name, "price": board.tiles[idx].price} for idx in offer_props]
        requested_prop_dicts = [{"index": idx, "name": board.tiles[idx].name, "price": board.tiles[idx].price} for idx in request_props]

        board_state = {
            "player_props": {p.player_id: [t.index for t in board.tiles if t.owner == p.player_id] for p in self.game.players}
        }

        is_pure_gift = (len(request_props) == 0 and request_cash == 0 and (len(offer_props) > 0 or offer_cash > 0))

        eval_res = self.llm_strategist.evaluate_trade(
            bot_id=target_id,
            bot_name=target.name,
            proposer_id=0,
            proposer_name=p0.name,
            offered_props=offered_prop_dicts,
            offered_cash=offer_cash,
            requested_props=requested_prop_dicts,
            requested_cash=request_cash,
            board_state=board_state,
            diplomatic_history=self.diplomatic_ledger.get(target_id, {}),
        )

        counter = eval_res.get("counter_offer")
        self.active_counter_offer = None

        if eval_res["accept"]:
            if p0.cash < offer_cash:
                eval_res["accept"] = False
                eval_res["reason"] = "Trade cancelled: Insufficient cash balance."
                return {
                    "accepted": False,
                    "reason": "Trade cancelled: Insufficient cash balance.",
                    "counter_offer": None,
                    "state": self.get_state(),
                }
            for idx in offer_props:
                board.tiles[idx].owner = target_id
                self.recent_trades.append((self.game.current_turn, idx))
            for idx in request_props:
                board.tiles[idx].owner = 0
                self.recent_trades.append((self.game.current_turn, idx))
            p0.cash = max(0, p0.cash - offer_cash)
            p0.cash += request_cash
            target.cash += offer_cash
            target.cash = max(0, target.cash - request_cash)

            self.match_stats[0]["trades_completed"] += 1
            self.match_stats[target_id]["trades_completed"] += 1

            # Update diplomatic memory
            if target_id not in self.diplomatic_ledger:
                self.diplomatic_ledger[target_id] = {"affinity": 0, "gifts_received": [], "favors_owed": 0, "trades_completed": 0}
            ledger = self.diplomatic_ledger[target_id]

            if is_pure_gift:
                gift_items = [board.tiles[idx].name for idx in offer_props]
                if offer_cash > 0:
                    gift_items.append(f"${offer_cash} cash")
                gift_str = ", ".join(gift_items)
                ledger["gifts_received"].append({"turn": self.game.current_turn, "gift": gift_str, "from_id": 0})
                ledger["affinity"] = min(100, ledger["affinity"] + 45)
                ledger["favors_owed"] += 1
                self.log(f"DIPLOMATIC MEMORY: {target.name} accepted your gift ({gift_str})! Affinity rose to {ledger['affinity']}/100.")
            else:
                ledger["affinity"] = min(100, ledger["affinity"] + 15)
                ledger["trades_completed"] += 1

            msg = f"TRADE ACCEPTED by {target.name}: {eval_res['reason']}"
            self.log(msg)
        else:
            msg = f"TRADE DECLINED by {target.name}: {eval_res['reason']}"
            self.log(msg)
            if counter:
                counter["bot_id"] = target_id
                counter["bot_name"] = target.name
                self.active_counter_offer = counter

        return {
            "accepted": eval_res["accept"],
            "reason": eval_res["reason"],
            "counter_offer": counter,
            "state": self.get_state(),
        }

    def accept_counter(self) -> dict[str, Any]:
        if not self.active_counter_offer:
            raise HTTPException(status_code=400, detail="No active counteroffer.")
        co = self.active_counter_offer
        p0 = self.game.players[0]
        target = self.game.players[co["bot_id"]]
        board = self.game.board

        if p0.cash < co["bot_wants_cash"]:
            raise HTTPException(status_code=400, detail="You do not have enough cash to accept this counteroffer.")

        for idx in co["bot_wants_props"]:
            board.tiles[idx].owner = co["bot_id"]
            self.recent_trades.append((self.game.current_turn, idx))
        for idx in co["bot_gives_props"]:
            board.tiles[idx].owner = 0
            self.recent_trades.append((self.game.current_turn, idx))

        p0.cash -= co["bot_wants_cash"]
        p0.cash += co["bot_gives_cash"]
        target.cash += co["bot_wants_cash"]
        target.cash -= co["bot_gives_cash"]

        # Record completed trade agreement in memory
        target_id = co["bot_id"]
        if target_id not in self.diplomatic_ledger:
            self.diplomatic_ledger[target_id] = {"affinity": 0, "gifts_received": [], "favors_owed": 0, "trades_completed": 0}
        self.diplomatic_ledger[target_id]["affinity"] = min(100, self.diplomatic_ledger[target_id]["affinity"] + 15)
        self.diplomatic_ledger[target_id]["trades_completed"] += 1
        self.match_stats[0]["trades_completed"] += 1
        self.match_stats[target_id]["trades_completed"] += 1

        self.log(f"COUNTEROFFER ACCEPTED! Executed trade agreement with {target.name}.")
        self.active_counter_offer = None
        return self.get_state()

    def accept_incoming_trade(self) -> dict[str, Any]:
        if not self.incoming_trade_proposal:
            return self.get_state()
        tp = self.incoming_trade_proposal
        p0 = self.game.players[0]
        bot = self.game.players[tp["bot_id"]]
        board = self.game.board

        # Transfer properties
        for idx in tp["bot_gives_props"]:
            board.tiles[idx].owner = 0
            self.recent_trades.append((self.game.current_turn, idx))
        for idx in tp["bot_wants_props"]:
            board.tiles[idx].owner = tp["bot_id"]
            self.recent_trades.append((self.game.current_turn, idx))

        # Transfer cash
        bot.cash -= tp["bot_gives_cash"]
        p0.cash += tp["bot_gives_cash"]

        bot_id = tp["bot_id"]
        if bot_id not in self.diplomatic_ledger:
            self.diplomatic_ledger[bot_id] = {"affinity": 0, "gifts_received": [], "favors_owed": 0, "trades_completed": 0}
        self.diplomatic_ledger[bot_id]["affinity"] = min(100, self.diplomatic_ledger[bot_id]["affinity"] + 25)
        self.diplomatic_ledger[bot_id]["trades_completed"] += 1
        self.match_stats[0]["trades_completed"] += 1
        self.match_stats[bot_id]["trades_completed"] += 1

        self.log(f"TRADE EXECUTED: Player 1 accepted trade with {bot.name}!")
        self.incoming_trade_proposal = None
        self._record_telemetry()
        return self.get_state()

    def decline_incoming_trade(self) -> dict[str, Any]:
        if not self.incoming_trade_proposal:
            return self.get_state()
        tp = self.incoming_trade_proposal
        bot = self.game.players[tp["bot_id"]]
        self.log(f"TRADE DECLINED: Player 1 declined trade offer from {bot.name}.")
        self.incoming_trade_proposal = None
        return self.get_state()

    def end_game(self) -> dict[str, Any]:
        self.game.game_over = True
        self._record_telemetry()
        board = self.game.board

        # Rank players: surviving players by net worth, bankrupt players by elimination order
        surviving = [p for p in self.game.players if not p.is_bankrupt]
        surviving.sort(key=lambda p: (p.net_worth(board), p.cash), reverse=True)

        bankrupt = [p for p in self.game.players if p.is_bankrupt]
        def elim_key(p):
            if p.player_id in self.game.elimination_order:
                return self.game.elimination_order.index(p.player_id)
            return -1
        # Reverse elimination order so last player eliminated gets highest rank among bankrupts
        bankrupt.sort(key=elim_key, reverse=True)

        ranked_players = surviving + bankrupt

        # Mario Party Superlatives with verified statistics
        superlatives = []
        # 1. Biggest Spender
        sp_id = max(range(4), key=lambda i: self.match_stats[i].get("cash_spent", 0))
        sp_val = self.match_stats[sp_id].get("cash_spent", 0)
        superlatives.append({
            "award": "Biggest Spender",
            "icon": "cash",
            "winner_id": sp_id,
            "winner_name": self.game.players[sp_id].name,
            "stat": f"${sp_val} Total Cash Spent",
            "desc": "Injected massive capital into deeds, houses, rent, and taxes."
        })

        # 2. Pawn Star & Mortgage King
        mg_id = max(range(4), key=lambda i: self.match_stats[i].get("mortgages_taken", 0))
        mg_val = self.match_stats[mg_id].get("mortgages_taken", 0)
        superlatives.append({
            "award": "Pawn Star & Mortgage King",
            "icon": "bank",
            "winner_id": mg_id,
            "winner_name": self.game.players[mg_id].name,
            "stat": f"{mg_val} Mortgages Taken",
            "desc": "Leveraged real-estate deeds for emergency liquidity."
        })

        # 3. Unluckiest Roller
        un_id = max(range(4), key=lambda i: self.match_stats[i].get("rent_paid", 0) + self.match_stats[i].get("taxes_paid", 0))
        un_val = self.match_stats[un_id].get("rent_paid", 0) + self.match_stats[un_id].get("taxes_paid", 0)
        superlatives.append({
            "award": "Unluckiest Roller",
            "icon": "hazard",
            "winner_id": un_id,
            "winner_name": self.game.players[un_id].name,
            "stat": f"${un_val} Paid in Rent & Taxes",
            "desc": "Fell victim to terrible dice probability and opponent rent traps."
        })

        # 4. Land Baron
        lb_id = max(range(4), key=lambda i: len([t for t in board.tiles if t.owner == i]))
        lb_val = len([t for t in board.tiles if t.owner == lb_id])
        superlatives.append({
            "award": "Land Baron",
            "icon": "castle",
            "winner_id": lb_id,
            "winner_name": self.game.players[lb_id].name,
            "stat": f"{lb_val} Properties Owned",
            "desc": "Controlled the largest expanse of Atlantic City territory."
        })

        # 5. Master Dealmaker
        dm_id = max(range(4), key=lambda i: self.match_stats[i].get("trades_completed", 0))
        dm_val = self.match_stats[dm_id].get("trades_completed", 0)
        superlatives.append({
            "award": "Master Dealmaker",
            "icon": "handshake",
            "winner_id": dm_id if dm_val > 0 else None,
            "winner_name": self.game.players[dm_id].name if dm_val > 0 else "None",
            "stat": f"{dm_val} Completed Deals",
            "desc": "Orchestrated diplomatic trades and reshaped the market." if dm_val > 0 else "No bilateral trade agreements were executed."
        })

        rankings = []
        debriefs = {}
        winner_name = ranked_players[0].name
        self.game.winner_id = ranked_players[0].player_id

        import concurrent.futures

        def _fetch_player_debrief(p_info):
            p_obj, r_idx, nw_val, props_cnt = p_info
            pid = p_obj.player_id
            pers = self.player_personalities.get(pid, "")
            d_text = self.llm_strategist.generate_debrief(
                bot_id=pid,
                bot_name=p_obj.name,
                rank=r_idx,
                total_players=len(self.game.players),
                winner_name=winner_name,
                final_cash=p_obj.cash,
                final_net_worth=nw_val,
                properties_count=props_cnt,
                personality=pers
            )
            return pid, d_text

        debrief_tasks = []
        for rank_idx, p in enumerate(ranked_players, start=1):
            pid = p.player_id
            p_props = len([t for t in board.tiles if t.owner == pid])
            p_nw = p.net_worth(board)
            rankings.append({
                "rank": rank_idx,
                "id": pid,
                "player_id": pid,
                "name": p.name,
                "cash": p.cash,
                "net_worth": p_nw,
                "properties": p_props,
                "properties_count": p_props,
                "is_bankrupt": p.is_bankrupt,
                "skin": self.player_skins.get(pid, pid),
                "type": self.player_types.get(pid, "bot"),
                "personality": self.player_personalities.get(pid, ""),
            })
            if pid != 0:
                debrief_tasks.append((p, rank_idx, p_nw, p_props))

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for pid, debrief in executor.map(_fetch_player_debrief, debrief_tasks):
                debriefs[pid] = debrief

        self.log(f"MATCH CONCLUDED! Winner: {winner_name} (Net Worth: ${ranked_players[0].net_worth(board)}). Standings and analytics generated.")

        return {
            "winner_id": self.game.winner_id,
            "winner_name": winner_name,
            "rankings": rankings,
            "superlatives": superlatives,
            "debriefs": debriefs,
            "telemetry": self.match_telemetry,
            "final_turn": self.game.current_turn,
            "state": self.get_state()
        }


session = GameSession()


@app.get("/")
def get_index():
    with open(os.path.join(VISUALIZER_DIR, "play.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/state")
def api_state():
    return session.get_state()


class SetupGameRequest(BaseModel):
    seed: Optional[int] = None
    max_turns: Optional[int] = 350
    starting_cash: Optional[int] = 1500
    randomize_turn_order: Optional[bool] = True
    players: Optional[list[dict]] = []

@app.post("/api/setup_game")
def api_setup_game(req: SetupGameRequest):
    return session.setup_game(req.dict())


@app.post("/api/roll")
def api_roll():
    return session.roll_dice()


@app.post("/api/buy")
def api_buy():
    return session.buy_property()


class AuctionRequest(BaseModel):
    human_bid: Optional[int] = 0

@app.post("/api/auction")
def api_auction(req: Optional[AuctionRequest] = None):
    bid = req.human_bid if req else 0
    return session.decline_auction(human_bid=bid)


class AuctionBidRequest(BaseModel):
    bid: int

@app.post("/api/submit_auction_bid")
def api_submit_auction_bid(req: AuctionBidRequest):
    return session.submit_auction_bid(req.bid)


class NegotiateRequest(BaseModel):
    message: str

@app.post("/api/negotiate_trade")
def api_negotiate_trade(req: NegotiateRequest):
    return session.negotiate_trade(req.message)


class BuildRequest(BaseModel):
    group: str

@app.post("/api/build")
def api_build(req: BuildRequest):
    return session.build_tier(req.group)


class PropRequest(BaseModel):
    index: int

@app.post("/api/mortgage")
def api_mortgage(req: PropRequest):
    return session.mortgage_property(req.index)

@app.post("/api/unmortgage")
def api_unmortgage(req: PropRequest):
    return session.unmortgage_property(req.index)


class JailRequest(BaseModel):
    action: str

@app.post("/api/jail")
def api_jail(req: JailRequest):
    return session.jail_action(req.action)


@app.post("/api/end_turn")
def api_end_turn():
    return session.end_turn()


@app.post("/api/start_ai")
def api_start_ai():
    return session.start_ai_turns()


@app.post("/api/step_ai")
def api_step_ai():
    return session.step_ai_turn()


class TradeRequest(BaseModel):
    target_id: int
    offer_props: list[int]
    offer_cash: int
    request_props: list[int]
    request_cash: int

@app.post("/api/trade")
def api_trade(req: TradeRequest):
    return session.handle_trade(
        req.target_id,
        req.offer_props,
        req.offer_cash,
        req.request_props,
        req.request_cash,
    )


@app.post("/api/accept_counter")
def api_accept_counter():
    return session.accept_counter()


@app.post("/api/accept_bot_trade")
def api_accept_bot_trade():
    return session.accept_incoming_trade()


@app.post("/api/decline_bot_trade")
def api_decline_bot_trade():
    return session.decline_incoming_trade()


@app.post("/api/end_game")
def api_end_game():
    return session.end_game()


class ChatRequest(BaseModel):
    bot_id: int
    message: str

@app.post("/api/chat")
def api_chat(req: ChatRequest):
    p0 = session.game.players[0]
    board = session.game.board

    # If bot_id <= 0, message is addressed to the entire Table (General Chat)
    if req.bot_id <= 0:
        active_bots = [p for p in session.game.players if p.player_id != 0 and not p.is_bankrupt]
        if not active_bots:
            active_bots = [p for p in session.game.players if p.player_id != 0]
        target = random.choice(active_bots)
        target_id = target.player_id
    else:
        if req.bot_id >= len(session.game.players):
            raise HTTPException(status_code=400, detail="Invalid bot ID.")
        target_id = req.bot_id
        target = session.game.players[target_id]

    diplomatic_info = session.diplomatic_ledger.get(target_id, {})

    all_players_info = [
        {
            "id": p.player_id,
            "name": p.name,
            "cash": p.cash,
            "net_worth": p.net_worth(board),
            "bankrupt": p.is_bankrupt,
        }
        for p in session.game.players
    ]

    context = {
        "human_name": p0.name,
        "human_cash": p0.cash,
        "human_net_worth": p0.net_worth(board),
        "bot_cash": target.cash,
        "bot_net_worth": target.net_worth(board),
        "turn": session.game.current_turn,
        "affinity": diplomatic_info.get("affinity", 0),
        "favors_owed": diplomatic_info.get("favors_owed", 0),
        "gifts": diplomatic_info.get("gifts_received", []),
        "trades_completed": diplomatic_info.get("trades_completed", 0),
        "all_players": all_players_info,
        "game_over": session.game.game_over,
        "winner_id": session.game.winner_id,
        "winner_name": session.game.players[session.game.winner_id].name if session.game.winner_id is not None else None,
        "bot_is_bankrupt": target.is_bankrupt,
        "human_is_bankrupt": p0.is_bankrupt,
    }
    personality = session.player_personalities.get(target_id, "")
    reply = session.llm_strategist.chat(target_id, target.name, req.message, context, personality=personality)
    return {
        "bot_id": target_id,
        "bot_name": target.name,
        "response": reply,
        "is_table": (req.bot_id <= 0)
    }



class OllamaConfigRequest(BaseModel):
    url: Optional[str] = None
    model: Optional[str] = None


@app.get("/api/ollama_status")
def api_ollama_status():
    return session.llm_strategist.get_status()


@app.post("/api/ollama_config")
def api_ollama_config(req: OllamaConfigRequest):
    return session.llm_strategist.set_config(req.url or "", req.model or "")


class BotChatterConfigRequest(BaseModel):
    chatter_rate: str


@app.post("/api/bot_chatter_config")
def api_bot_chatter_config(req: BotChatterConfigRequest):
    rate = req.chatter_rate.lower().strip()
    if rate in ["high", "normal", "low", "silent"]:
        session.bot_chatter_rate = rate
    return {"chatter_rate": session.bot_chatter_rate}


@app.post("/api/fast_forward")
def api_fast_forward():
    steps = 0
    while not session.game.game_over and steps < 60:
        session.game.execute_turn()
        steps += 1
    return session.get_state()


class ResetRequest(BaseModel):
    seed: Optional[int] = None

@app.post("/api/reset")
def api_reset(req: ResetRequest):
    session.reset(req.seed)
    return session.get_state()


def main():
    port = 8000
    print(f"\n=======================================================")
    print(f"Monopoly AI Arena: Play Against PPO-Agent & Bots")
    print(f"Open your browser at: http://localhost:{port}")
    print(f"=======================================================\n")
    try:
        webbrowser.open(f"http://localhost:{port}")
    except Exception:
        pass
    uvicorn.run("play_server:app", host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()
