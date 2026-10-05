"""
High-Speed Official Monopoly Tournament Game Engine.
Audited against Hasbro official rules and DARPA GNOME schema.
"""

from typing import Optional, Callable
import random

from monopoly_core.constants import (
    TileType,
    ColorGroup,
    GO_SALARY,
    JAIL_FINE,
    INCOME_TAX,
    LUXURY_TAX,
    COLOR_GROUP_TILES,
)
from monopoly_core.board import Board, Tile
from monopoly_core.player import Player
from monopoly_core.cards import Card, CardAction
from monopoly_core.auction import conduct_auction


class MonopolyGame:
    def __init__(
        self,
        num_players: int = 4,
        seed: Optional[int] = None,
        max_turns: int = 500,
        enable_logging: bool = False,
    ):
        self.rng = random.Random(seed)
        self.max_turns = max_turns
        self.enable_logging = enable_logging
        self.log_messages: list[str] = []

        self.board = Board(self.rng)
        self.players = [
            Player(player_id=i, name=f"Player_{i+1}") for i in range(num_players)
        ]
        self.current_player_idx = 0
        self.current_turn = 0
        self.game_over = False
        self.winner_id: Optional[int] = None
        self.last_dice: tuple[int, int] = (1, 1)
        self.last_card_drawn: Optional[dict[str, Any]] = None
        self.last_jail_event: Optional[dict[str, Any]] = None
        self.auction_handler: Optional[Callable] = None
        self.interactive_human: bool = False
        self.elimination_order: list[int] = []
        self.turn_order: list[int] = list(range(num_players))
        self.turn_order_pos: int = 0

    def log(self, message: str):
        if self.enable_logging:
            self.log_messages.append(message)

    @property
    def active_players(self) -> list[Player]:
        return [p for p in self.players if not p.is_bankrupt]

    def roll_dice(self) -> tuple[int, int, bool]:
        d1 = self.rng.randint(1, 6)
        d2 = self.rng.randint(1, 6)
        self.last_dice = (d1, d2)
        return d1, d2, (d1 == d2)

    def handle_payment(self, debtor: Player, creditor: Optional[Player], amount: int) -> bool:
        """
        Handles payment from debtor to creditor (or bank if creditor is None).
        Liquidates assets if needed.
        Returns True if paid, False if bankrupt.
        """
        if debtor.cash < amount:
            pre_mortgaged = {t.index: t.is_mortgaged for t in self.board.tiles if t.owner == debtor.player_id}
            can_pay = debtor.liquidate_assets_to_cover(amount, self.board)
            new_mortgages = [t for t in self.board.tiles if t.owner == debtor.player_id and not pre_mortgaged.get(t.index, False) and t.is_mortgaged]
            for t in new_mortgages:
                self.log(f"EMERGENCY MORTGAGE: {debtor.name} mortgaged {t.name} for +${t.mortgage_value} to cover debt.")
            if not can_pay:
                # Bankruptcy!
                debtor.is_bankrupt = True
                if debtor.player_id not in self.elimination_order:
                    self.elimination_order.append(debtor.player_id)
                self.log(f"{debtor.name} went BANKRUPT to {'Bank' if creditor is None else creditor.name}!")
                
                # Transfer remaining assets
                if creditor is not None:
                    creditor.cash += debtor.cash
                    debtor.cash = 0
                    transferred = []
                    for tile in self.board.tiles:
                        if tile.owner == debtor.player_id:
                            tile.owner = creditor.player_id
                            status = "MORTGAGED" if tile.is_mortgaged else "Clean"
                            transferred.append(f"{tile.name} ({status})")
                    if transferred:
                        self.log(f"ASSET TRANSFER: {creditor.name} acquired {len(transferred)} properties from {debtor.name}: {', '.join(transferred)}.")
                else:
                    # Bankrupt to bank (taxes/cards): properties are unmortgaged and auctioned
                    for tile in self.board.tiles:
                        if tile.owner == debtor.player_id:
                            tile.owner = None
                            tile.is_mortgaged = False
                            tile.num_houses = 0
                            tile.num_hotels = 0
                            # Auction unowned property
                            conduct_auction(tile, self.players, self.board)
                return False

        debtor.cash -= amount
        if creditor is not None:
            creditor.cash += amount
        return True

    def send_to_jail(self, player: Player):
        self.log(f"{player.name} is sent to JAIL!")
        self.last_jail_event = {"player_id": player.player_id, "player_name": player.name}
        player.position = 10
        player.in_jail = True
        player.jail_turns = 0
        player.consecutive_doubles = 0

    def resolve_card_effect(
        self,
        player: Player,
        card: Card,
        dice_sum: int,
    ):
        self.log(f"{player.name} drew card: '{card.name}'")

        if card.action == CardAction.ADVANCE_TO:
            dest = card.destination
            assert dest is not None
            if dest < player.position and dest != 10:  # Passed Go (unless sent to jail)
                player.cash += GO_SALARY
                self.log(f"{player.name} collected ${GO_SALARY} for passing Go.")
            if self.last_card_drawn:
                self.last_card_drawn["destination"] = dest
                self.last_card_drawn["destination_name"] = self.board.tiles[dest].name
            player.position = dest
            self.resolve_tile_landing(player, dice_sum)

        elif card.action == CardAction.ADVANCE_TO_NEAREST_UTILITY:
            # Utilities are at 12 and 28
            dest = 12 if (player.position > 28 or player.position <= 12) else 28
            if dest < player.position:
                player.cash += GO_SALARY
            if self.last_card_drawn:
                self.last_card_drawn["destination"] = dest
                self.last_card_drawn["destination_name"] = self.board.tiles[dest].name
            player.position = dest
            self.resolve_tile_landing(player, dice_sum, forced_utility_multiplier=10)

        elif card.action == CardAction.ADVANCE_TO_NEAREST_RAILROAD:
            # Railroads at 5, 15, 25, 35
            pos = player.position
            if pos <= 5 or pos > 35:
                dest = 5
            elif pos <= 15:
                dest = 15
            elif pos <= 25:
                dest = 25
            else:
                dest = 35
            if dest < player.position:
                player.cash += GO_SALARY
            if self.last_card_drawn:
                self.last_card_drawn["destination"] = dest
                self.last_card_drawn["destination_name"] = self.board.tiles[dest].name
            player.position = dest
            self.resolve_tile_landing(player, dice_sum, force_double_railroad=True)

        elif card.action == CardAction.GO_BACK_SPACES:
            dest = (player.position - card.amount) % 40
            self.log(f"{player.name} moved back {card.amount} spaces to {self.board.tiles[dest].name}.")
            if self.last_card_drawn:
                self.last_card_drawn["destination"] = dest
                self.last_card_drawn["destination_name"] = self.board.tiles[dest].name
            player.position = dest
            self.resolve_tile_landing(player, dice_sum)

        elif card.action == CardAction.GO_TO_JAIL:
            self.send_to_jail(player)

        elif card.action == CardAction.GET_OUT_OF_JAIL_FREE:
            player.get_out_of_jail_cards += 1
            self.log(f"{player.name} acquired a Get Out of Jail Free card!")

        elif card.action == CardAction.CASH_FROM_BANK:
            player.cash += card.amount
            self.log(f"{player.name} received ${card.amount} from the Bank. (Cash: ${player.cash})")

        elif card.action == CardAction.PAY_BANK:
            self.log(f"{player.name} paid ${card.amount} to the Bank for '{card.name}'.")
            self.handle_payment(player, None, card.amount)

        elif card.action == CardAction.COLLECT_FROM_PLAYERS:
            self.log(f"{player.name} collected ${card.amount} from each active player.")
            for other in self.players:
                if other.player_id != player.player_id and not other.is_bankrupt:
                    self.handle_payment(other, player, card.amount)

        elif card.action == CardAction.PAY_PLAYERS:
            self.log(f"{player.name} paid ${card.amount} to each active player.")
            for other in self.players:
                if other.player_id != player.player_id and not other.is_bankrupt:
                    if not self.handle_payment(player, other, card.amount):
                        break

        elif card.action == CardAction.PROPERTY_ASSESSMENT:
            total_bill = 0
            for tile in self.board.tiles:
                if tile.owner == player.player_id:
                    total_bill += tile.num_houses * card.house_cost + tile.num_hotels * card.hotel_cost
            if total_bill > 0:
                self.log(f"{player.name} assessed repairs: Paid ${total_bill} (${card.house_cost}/house, ${card.hotel_cost}/hotel).")
                self.handle_payment(player, None, total_bill)
            else:
                self.log(f"{player.name} has no developed properties. Assessment bill: $0.")

    def resolve_tile_landing(
        self,
        player: Player,
        dice_sum: int,
        forced_utility_multiplier: Optional[int] = None,
        force_double_railroad: bool = False,
    ):
        if player.is_bankrupt:
            return

        tile = self.board.tiles[player.position]
        self.log(f"{player.name} landed on {tile.name} ({tile.index}).")

        # 1. Special non-property tiles
        if tile.tile_type == TileType.SPECIAL:
            if tile.index == 30:  # Go To Jail
                self.send_to_jail(player)
            elif tile.name.startswith("Chance"):
                card = self.board.chance_deck.draw()
                self.last_card_drawn = {
                    "deck": "CHANCE",
                    "name": card.name,
                    "action": card.action.name,
                    "amount": card.amount,
                    "player_id": player.player_id,
                    "player_name": player.name,
                    "card_drawn_at": player.position,
                }
                self.resolve_card_effect(player, card, dice_sum)
                if card.action != CardAction.GET_OUT_OF_JAIL_FREE:
                    self.board.chance_deck.return_card(card)
            elif tile.name.startswith("Community Chest"):
                card = self.board.community_chest_deck.draw()
                self.last_card_drawn = {
                    "deck": "COMMUNITY CHEST",
                    "name": card.name,
                    "action": card.action.name,
                    "amount": card.amount,
                    "player_id": player.player_id,
                    "player_name": player.name,
                    "card_drawn_at": player.position,
                }
                self.resolve_card_effect(player, card, dice_sum)
                if card.action != CardAction.GET_OUT_OF_JAIL_FREE:
                    self.board.community_chest_deck.return_card(card)

        # 2. Tax tiles
        elif tile.tile_type == TileType.TAX:
            tax = INCOME_TAX if tile.index == 4 else LUXURY_TAX
            tax_name = "Income Tax ($200)" if tile.index == 4 else "Luxury Tax ($100)"
            if self.handle_payment(player, None, tax):
                self.log(f"{player.name} paid {tax_name} to the Bank. (Cash: ${player.cash})")

        # 3. Purchasable properties
        elif tile.is_purchasable:
            if tile.owner is None:
                # If interactive human player, defer purchase/auction decision to session controller
                if player.agent is None and self.interactive_human:
                    return

                # Decision delegated to agent if present
                if hasattr(player.agent, "on_buy_decision"):
                    wants_to_buy = player.agent.on_buy_decision(player, tile, self)
                else:
                    wants_to_buy = (player.cash >= tile.price + 100)

                if wants_to_buy and player.cash >= tile.price:
                    player.cash -= tile.price
                    tile.owner = player.player_id
                    self.log(f"{player.name} bought {tile.name} for ${tile.price}.")
                else:
                    self.log(f"{player.name} declined to buy {tile.name}. Triggering AUCTION.")
                    if self.auction_handler is not None:
                        winner_id, winning_bid = self.auction_handler(tile)
                    else:
                        winner_id, winning_bid = conduct_auction(tile, self.players, self.board)
                    if winner_id is not None:
                        self.log(f"AUCTION: {self.players[winner_id].name} won {tile.name} for ${winning_bid}.")
            elif tile.owner != player.player_id:
                # Owned by another player: Rent is due
                rent = self.board.calculate_rent(
                    tile.index,
                    dice_sum,
                    forced_utility_multiplier=forced_utility_multiplier,
                    force_double_railroad=force_double_railroad,
                )
                if rent > 0:
                    creditor = self.players[tile.owner]
                    self.log(f"{player.name} owes ${rent} rent to {creditor.name} for {tile.name}.")
                    self.handle_payment(player, creditor, rent)

    def execute_turn(self) -> bool:
        """
        Executes a complete turn for the current player.
        Returns True if game is still active, False if game ended.
        """
        if self.game_over:
            return False

        player = self.players[self.current_player_idx]
        if player.is_bankrupt:
            self._advance_turn()
            return not self.game_over

        self.last_card_drawn = None
        self.last_jail_event = None
        self.log(f"\n--- {player.name}'s turn (Turn {self.current_turn}, Cash: ${player.cash}) ---")

        # 1. Pre-Roll Building & Development (delegated to agent if present)
        if hasattr(player.agent, "on_pre_roll"):
            player.agent.on_pre_roll(player, self)
        else:
            self._ai_build_houses(player)
            self._ai_unmortgage_properties(player)

        # 2. Jail Handling
        if player.in_jail:
            player.jail_turns += 1
            if hasattr(player.agent, "on_jail_decision"):
                player.agent.on_jail_decision(player, self)
            else:
                if player.get_out_of_jail_cards > 0:
                    player.get_out_of_jail_cards -= 1
                    player.in_jail = False
                    player.jail_turns = 0
                    self.log(f"{player.name} used a Get Out of Jail Free card!")
                elif player.cash >= 300:
                    player.cash -= JAIL_FINE
                    player.in_jail = False
                    player.jail_turns = 0
                    self.log(f"{player.name} paid ${JAIL_FINE} fine to exit jail.")

        # 3. Roll Dice
        d1, d2, is_double = self.roll_dice()
        dice_sum = d1 + d2
        self.log(f"{player.name} rolled {d1} + {d2} = {dice_sum} ({'DOUBLES' if is_double else 'not doubles'}).")

        if player.in_jail:
            if is_double:
                player.in_jail = False
                player.jail_turns = 0
                self.log(f"{player.name} rolled doubles to exit jail!")
                player.position = (player.position + dice_sum) % 40
                self.resolve_tile_landing(player, dice_sum)
            elif player.jail_turns >= 3:
                # Must pay $50 on 3rd turn
                self.log(f"{player.name} spent 3 turns in jail, must pay ${JAIL_FINE}.")
                if self.handle_payment(player, None, JAIL_FINE):
                    player.in_jail = False
                    player.jail_turns = 0
                    player.position = (player.position + dice_sum) % 40
                    self.resolve_tile_landing(player, dice_sum)
            else:
                self.log(f"{player.name} remains in jail.")
            
            self._check_game_over()
            self._advance_turn()
            return not self.game_over

        # 4. Normal Movement
        if is_double:
            player.consecutive_doubles += 1
            if player.consecutive_doubles >= 3:
                self.log(f"{player.name} rolled 3 DOUBLES in a row! Sent to Jail!")
                self.send_to_jail(player)
                self._check_game_over()
                self._advance_turn()
                return not self.game_over
        else:
            player.consecutive_doubles = 0

        new_pos = (player.position + dice_sum) % 40
        if new_pos < player.position:
            # Passed Go
            player.cash += GO_SALARY
            self.log(f"{player.name} collected ${GO_SALARY} for passing Go.")
        player.position = new_pos

        self.resolve_tile_landing(player, dice_sum)

        self._check_game_over()

        # If rolled doubles and not in jail and not bankrupt -> player gets another turn
        if is_double and not player.in_jail and not player.is_bankrupt:
            self.log(f"{player.name} gets another roll for doubles!")
            return not self.game_over

        self._advance_turn()
        return not self.game_over

    def _ai_build_houses(self, player: Player):
        """Builds houses evenly on monopolies if cash > $300."""
        while player.cash >= 300:
            built_any = False
            for tile in sorted(
                [t for t in self.board.tiles if t.owner == player.player_id and t.tile_type == TileType.STREET],
                key=lambda t: t.house_cost,
            ):
                if self.board.can_build_house(tile.index, player.player_id) and player.cash >= tile.house_cost + 200:
                    player.cash -= tile.house_cost
                    if tile.num_houses == 4:
                        tile.num_houses = 0
                        tile.num_hotels = 1
                        self.board.available_houses += 4
                        self.board.available_hotels -= 1
                        self.log(f"{player.name} built a HOTEL on {tile.name}.")
                    else:
                        tile.num_houses += 1
                        self.board.available_houses -= 1
                        self.log(f"{player.name} built house #{tile.num_houses} on {tile.name}.")
                    built_any = True
                    break
            if not built_any:
                break

    def _ai_unmortgage_properties(self, player: Player):
        """Unmortgages properties if cash is abundant (> $400)."""
        if player.cash < 400:
            return
        for tile in sorted(
            [t for t in self.board.tiles if t.owner == player.player_id and t.is_mortgaged],
            key=lambda t: t.mortgage_value,
        ):
            cost = int(tile.mortgage_value * 1.1)
            if player.cash >= cost + 200:
                player.cash -= cost
                tile.is_mortgaged = False
                self.log(f"{player.name} unmortgaged {tile.name} for ${cost}.")

    def _advance_turn(self):
        self.current_turn += 1
        if hasattr(self, "turn_order") and self.turn_order:
            self.turn_order_pos = (self.turn_order_pos + 1) % len(self.turn_order)
            self.current_player_idx = self.turn_order[self.turn_order_pos]
        else:
            self.current_player_idx = (self.current_player_idx + 1) % len(self.players)
        if self.current_turn >= self.max_turns:
            self._end_by_turn_limit()

    def _check_game_over(self):
        if self.game_over:
            return
        active = self.active_players
        if len(active) <= 1:
            self.game_over = True
            if len(active) == 1:
                self.winner_id = active[0].player_id
                self.log(f"\n==========================================")
                self.log(f"GAME OVER! {active[0].name} WINS!")
                self.log(f"==========================================")

    def _end_by_turn_limit(self):
        self.game_over = True
        active = self.active_players
        if active:
            # Highest net worth wins if time expires
            ranked = sorted(active, key=lambda p: p.net_worth(self.board), reverse=True)
            self.winner_id = ranked[0].player_id
            self.log(f"Turn limit reached. {ranked[0].name} wins by net worth (${ranked[0].net_worth(self.board)}).")

    def play_full_game(self) -> Optional[int]:
        while not self.game_over:
            self.execute_turn()
        return self.winner_id
