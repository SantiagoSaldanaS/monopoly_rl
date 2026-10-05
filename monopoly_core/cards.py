"""
Chance and Community Chest Decks.
Audited against Hasbro official rules and DARPA GNOME schema.
"""

from enum import Enum, auto
from typing import NamedTuple, Optional, Callable
import random

class CardAction(Enum):
    ADVANCE_TO = auto()
    ADVANCE_TO_NEAREST_UTILITY = auto()
    ADVANCE_TO_NEAREST_RAILROAD = auto()
    GO_BACK_SPACES = auto()
    GO_TO_JAIL = auto()
    GET_OUT_OF_JAIL_FREE = auto()
    CASH_FROM_BANK = auto()
    PAY_BANK = auto()
    COLLECT_FROM_PLAYERS = auto()
    PAY_PLAYERS = auto()
    PROPERTY_ASSESSMENT = auto()

class Card(NamedTuple):
    name: str
    action: CardAction
    destination: Optional[int] = None
    amount: int = 0
    house_cost: int = 0
    hotel_cost: int = 0

OFFICIAL_CHANCE_CARDS = [
    Card("Advance to Go (Collect $200)", CardAction.ADVANCE_TO, destination=0),
    Card("Advance to Illinois Avenue", CardAction.ADVANCE_TO, destination=24),
    Card("Advance to St. Charles Place", CardAction.ADVANCE_TO, destination=11),
    Card("Advance token to nearest Utility", CardAction.ADVANCE_TO_NEAREST_UTILITY),
    Card("Advance token to nearest Railroad (1)", CardAction.ADVANCE_TO_NEAREST_RAILROAD),
    Card("Advance token to nearest Railroad (2)", CardAction.ADVANCE_TO_NEAREST_RAILROAD),
    Card("Bank pays you dividend of $50", CardAction.CASH_FROM_BANK, amount=50),
    Card("Get Out of Jail Free", CardAction.GET_OUT_OF_JAIL_FREE),
    Card("Go Back 3 Spaces", CardAction.GO_BACK_SPACES, amount=3),
    Card("Go to Jail", CardAction.GO_TO_JAIL),
    Card("Make general repairs on all your property", CardAction.PROPERTY_ASSESSMENT, house_cost=25, hotel_cost=100),
    Card("Pay poor tax of $15", CardAction.PAY_BANK, amount=15),
    Card("Take a trip to Reading Railroad", CardAction.ADVANCE_TO, destination=5),
    Card("Take a walk on the Boardwalk", CardAction.ADVANCE_TO, destination=39),
    Card("You have been elected Chairman of the Board. Pay each player $50", CardAction.PAY_PLAYERS, amount=50),
    Card("Your building loan matures. Collect $150", CardAction.CASH_FROM_BANK, amount=150),
    Card("You have won a crossword competition. Collect $100", CardAction.CASH_FROM_BANK, amount=100),
]

OFFICIAL_COMMUNITY_CHEST_CARDS = [
    Card("Advance to Go (Collect $200)", CardAction.ADVANCE_TO, destination=0),
    Card("Bank error in your favor. Collect $200", CardAction.CASH_FROM_BANK, amount=200),
    Card("Doctor's fee. Pay $50", CardAction.PAY_BANK, amount=50),
    Card("From sale of stock you get $50", CardAction.CASH_FROM_BANK, amount=50),
    Card("Get Out of Jail Free", CardAction.GET_OUT_OF_JAIL_FREE),
    Card("Go to Jail", CardAction.GO_TO_JAIL),
    Card("Holiday fund matures. Receive $100", CardAction.CASH_FROM_BANK, amount=100),
    Card("Income tax refund. Collect $20", CardAction.CASH_FROM_BANK, amount=20),
    Card("It is your birthday. Collect $10 from every player", CardAction.COLLECT_FROM_PLAYERS, amount=10),
    Card("Life insurance matures. Collect $100", CardAction.CASH_FROM_BANK, amount=100),
    Card("Pay hospital fees of $50", CardAction.PAY_BANK, amount=50),
    Card("Pay school fees of $50", CardAction.PAY_BANK, amount=50),
    Card("Receive $25 consultancy fee", CardAction.CASH_FROM_BANK, amount=25),
    Card("You are assessed for street repairs", CardAction.PROPERTY_ASSESSMENT, house_cost=40, hotel_cost=115),
    Card("You have won second prize in a beauty contest. Collect $10", CardAction.CASH_FROM_BANK, amount=10),
    Card("You inherit $100", CardAction.CASH_FROM_BANK, amount=100),
]

class CardDeck:
    def __init__(self, cards: list[Card], rng: random.Random):
        self.cards = list(cards)
        self.rng = rng
        self.rng.shuffle(self.cards)
        self.discard_pile: list[Card] = []

    def draw(self) -> Card:
        if not self.cards:
            self.cards = self.discard_pile
            self.discard_pile = []
            self.rng.shuffle(self.cards)
        card = self.cards.pop(0)
        # Note: Get Out of Jail Free card is not immediately discarded until used
        return card

    def return_card(self, card: Card):
        self.discard_pile.append(card)
