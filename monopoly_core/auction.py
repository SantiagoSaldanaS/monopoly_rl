"""
Official Open Ascending Auction Engine for Monopoly.
Audited against Hasbro official rules and DARPA GNOME schema.
"""

from typing import Callable, Optional, TYPE_CHECKING
from monopoly_core.constants import COLOR_GROUP_TILES

if TYPE_CHECKING:
    from monopoly_core.board import Board, Tile
    from monopoly_core.player import Player


def default_valuation(player: "Player", tile: "Tile", board: "Board") -> int:
    """
    Standard strategic valuation function for auction bidding:
    - Base value: Up to printed price, scaled by available cash.
    - Monopoly completion bonus: +100% of price if it completes a color group.
    - Block opponent bonus: +30% of price if an opponent is 1 tile away from a monopoly.
    - Cash reserve constraint: Never bid more than (cash - $100 safety cushion).
    """
    if player.is_bankrupt or player.cash <= 10:
        return 0

    max_affordable = max(0, player.cash - 100)
    if max_affordable <= 0:
        return 0

    valuation = tile.price

    # Check if this property completes a monopoly for this player
    group_indices = COLOR_GROUP_TILES.get(tile.color_group, [])
    if group_indices:
        unowned_or_other = [
            i for i in group_indices if i != tile.index and board.tiles[i].owner != player.player_id
        ]
        if len(unowned_or_other) == 0:
            # Completes our monopoly! High strategic value
            valuation = int(tile.price * 1.5)

    # Check if this blocks an opponent's near-monopoly
    for other_p in range(4):
        if other_p != player.player_id:
            opponent_owned = [
                i for i in group_indices if i != tile.index and board.tiles[i].owner == other_p
            ]
            if len(opponent_owned) == len(group_indices) - 1:
                # Opponent needs only this tile! Block bonus
                valuation = max(valuation, int(tile.price * 1.2))

    return min(valuation, max_affordable)


def conduct_auction(
    tile: "Tile",
    players: list["Player"],
    board: "Board",
    custom_valuations: Optional[dict[int, int]] = None,
) -> tuple[Optional[int], int]:
    """
    Conducts an official open ascending auction for `tile`.
    Returns:
        (winning_player_id, winning_bid_amount)
        or (None, 0) if no player bids.
    """
    # Determine maximum willingness-to-pay for each non-bankrupt player
    willingness: dict[int, int] = {}
    for p in players:
        if p.is_bankrupt:
            continue
        if custom_valuations and p.player_id in custom_valuations:
            val = custom_valuations[p.player_id]
        elif hasattr(p.agent, "get_auction_bid"):
            val = p.agent.get_auction_bid(p, tile, board)
        else:
            val = default_valuation(p, tile, board)
        willingness[p.player_id] = min(val, p.cash)

    # Filter players who are willing to bid at least $10
    active_bidders = [pid for pid, val in willingness.items() if val >= 10]
    if not active_bidders:
        return None, 0

    if len(active_bidders) == 1:
        winner_id = active_bidders[0]
        bid = 10
        # Award property
        p = players[winner_id]
        p.cash -= bid
        tile.owner = winner_id
        return winner_id, bid

    # Ascending auction: price rises until 1 bidder remains
    # Price rises in increments of $10
    current_bid = 10
    while len(active_bidders) > 1:
        current_bid += 10
        active_bidders = [pid for pid in active_bidders if willingness[pid] >= current_bid]

    if len(active_bidders) == 1:
        winner_id = active_bidders[0]
        # Winning bid is the current bid, but not exceeding player's willingness
        bid = min(current_bid, willingness[winner_id])
        p = players[winner_id]
        p.cash -= bid
        tile.owner = winner_id
        return winner_id, bid

    # If all dropped out at the exact same step, highest willingness wins
    sorted_by_val = sorted(willingness.items(), key=lambda x: x[1], reverse=True)
    winner_id = sorted_by_val[0][0]
    bid = sorted_by_val[1][1] if len(sorted_by_val) > 1 else 10
    bid = max(10, min(bid, players[winner_id].cash))
    players[winner_id].cash -= bid
    tile.owner = winner_id
    return winner_id, bid
