"""
Observation State Encoder for Monopoly Reinforcement Learning.
Encodes the complete board, player, and bank state into a normalized NumPy vector.
Provides clean discrete ownership features and explicit monopoly group awareness.
"""

import numpy as np
from typing import TYPE_CHECKING
from monopoly_core.constants import ALL_PROPERTY_INDICES, MAX_HOUSES, MAX_HOTELS, COLOR_GROUP_TILES, ColorGroup

if TYPE_CHECKING:
    from monopoly_core.game import MonopolyGame

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

OBSERVATION_DIMENSION = 439

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


def encode_observation(game: "MonopolyGame", current_player_id: int) -> np.ndarray:
    """
    Encodes the game state from the perspective of `current_player_id`.
    Returns a 1D float32 numpy array of shape (439,).

    Structure:
    1. Property States (28 properties * 8 features = 224 floats):
       - Is unowned (1.0 or 0.0)
       - Owned by self (1.0 or 0.0)
       - Owned by opponent 1 relative (1.0 or 0.0)
       - Owned by opponent 2 relative (1.0 or 0.0)
       - Owned by opponent 3 relative (1.0 or 0.0)
       - Houses: num_houses / 4.0
       - Hotels: 1.0 if hotel else 0.0
       - Mortgaged: 1.0 if mortgaged else 0.0
    2. Monopoly Group Summary (8 groups * 2 features = 16 floats):
       - Self owns complete monopoly on group (1.0 or 0.0)
       - Opponent owns complete monopoly on group (1.0 or 0.0)
    3. Bank Inventory (2 floats):
       - Available houses: houses / 32.0
       - Available hotels: hotels / 12.0
    4. Players States (4 players * 46 features = 184 floats):
       - Cash normalized: tanh(cash / 1000.0)
       - Position one-hot (40 floats)
       - In jail (1.0 or 0.0)
       - Jail turns / 3.0
       - Get out of jail cards / 2.0
       - Net worth normalized: tanh(net_worth / 2000.0)
       - Is bankrupt (1.0 or 0.0)
    5. Global Turn Stats (6 floats):
       - Current turn / max_turns
       - Is active player turn (1.0 or 0.0)
       - Relative player rank by net worth (one-hot 4 floats)
    6. Perceptual Sensory & Decision Features (7 floats):
       - Current landed tile price / 400.0
       - Current landed tile is unowned (1.0 or 0.0)
       - Current tile completes self monopoly (1.0 or 0.0)
       - Current tile blocks opponent monopoly (1.0 or 0.0)
       - Current tile Markov landing probability
       - Remaining unowned purchasable properties / 28.0
       - Impending danger zone threat (tanh(max rent in 2..12 roll ahead / 500.0))

    Total size: 224 + 16 + 2 + 184 + 6 + 7 = 439 floats.
    """
    obs = []
    board = game.board

    # 1. Properties (28 * 8 = 224)
    for idx in ALL_PROPERTY_INDICES:
        tile = board.tiles[idx]
        if tile.owner is None:
            obs.extend([1.0, 0.0, 0.0, 0.0, 0.0])
        else:
            rel = (tile.owner - current_player_id) % 4
            owner_vec = [0.0, 0.0, 0.0, 0.0, 0.0]
            # owner_vec[0] is unowned, 1 is self, 2 is opp1, 3 is opp2, 4 is opp3
            owner_vec[rel + 1] = 1.0
            obs.extend(owner_vec)

        obs.append(tile.num_houses / 4.0)
        obs.append(1.0 if tile.num_hotels == 1 else 0.0)
        obs.append(1.0 if tile.is_mortgaged else 0.0)

    # 2. Monopoly Group Summary (8 * 2 = 16)
    for group in BUILDABLE_GROUPS:
        self_owns = 1.0 if board.owns_full_group(current_player_id, group) else 0.0
        opp_owns = 0.0
        for off in (1, 2, 3):
            opp_id = (current_player_id + off) % 4
            if board.owns_full_group(opp_id, group):
                opp_owns = 1.0
                break
        obs.append(self_owns)
        obs.append(opp_owns)

    # 3. Bank Inventory (2)
    obs.append(board.available_houses / float(MAX_HOUSES))
    obs.append(board.available_hotels / float(MAX_HOTELS))

    # 4. Player States (ordered relative to current player: self, opp1, opp2, opp3) (4 * 46 = 184)
    for offset in range(4):
        p_id = (current_player_id + offset) % 4
        p = game.players[p_id]

        obs.append(float(np.tanh(p.cash / 1000.0)))

        # Position one-hot (40)
        pos_one_hot = [0.0] * 40
        if 0 <= p.position < 40:
            pos_one_hot[p.position] = 1.0
        obs.extend(pos_one_hot)

        obs.append(1.0 if p.in_jail else 0.0)
        obs.append(p.jail_turns / 3.0)
        obs.append(p.get_out_of_jail_cards / 2.0)
        obs.append(float(np.tanh(p.net_worth(board) / 2000.0)))
        obs.append(1.0 if p.is_bankrupt else 0.0)

    # 5. Turn Stats (6)
    obs.append(game.current_turn / float(game.max_turns))
    obs.append(1.0 if game.current_player_idx == current_player_id else 0.0)

    # Rank by net worth
    active = [p for p in game.players if not p.is_bankrupt]
    active_sorted = sorted(active, key=lambda p: p.net_worth(board), reverse=True)
    rank = 3
    for r, p in enumerate(active_sorted):
        if p.player_id == current_player_id:
            rank = r
            break
    rank_one_hot = [0.0] * 4
    rank_one_hot[rank] = 1.0
    obs.extend(rank_one_hot)

    # 6. Perceptual Sensory & Decision Features (7)
    cur_pos = game.players[current_player_id].position
    current_tile = board.tiles[cur_pos]

    # Current tile features
    if current_tile.is_purchasable:
        obs.append(float(current_tile.price) / 400.0)
        obs.append(1.0 if current_tile.owner is None else 0.0)

        # Completes self monopoly?
        completes_self = 0.0
        if current_tile.color_group is not None:
            indices = COLOR_GROUP_TILES.get(current_tile.color_group, [])
            unowned_or_other = [i for i in indices if i != current_tile.index and board.tiles[i].owner != current_player_id]
            if len(unowned_or_other) == 0:
                completes_self = 1.0
        obs.append(completes_self)

        # Blocks opponent monopoly?
        blocks_opp = 0.0
        if current_tile.color_group is not None:
            indices = COLOR_GROUP_TILES.get(current_tile.color_group, [])
            for off in (1, 2, 3):
                opp_id = (current_player_id + off) % 4
                opp_count = sum(1 for i in indices if i != current_tile.index and board.tiles[i].owner == opp_id)
                if len(indices) > 0 and opp_count == len(indices) - 1:
                    blocks_opp = 1.0
                    break
        obs.append(blocks_opp)

        obs.append(float(MARKOV_LANDING_PROBS.get(cur_pos, 0.025)))
    else:
        obs.extend([0.0, 0.0, 0.0, 0.0, float(MARKOV_LANDING_PROBS.get(cur_pos, 0.025))])

    # Unowned properties remaining on the board
    unowned_count = sum(1 for t in board.tiles if t.is_purchasable and t.owner is None)
    obs.append(unowned_count / 28.0)

    # Impending danger zone threat (maximum rent within next 2-12 roll)
    max_threat = 0.0
    for roll in range(2, 13):
        target_pos = (cur_pos + roll) % 40
        target_tile = board.tiles[target_pos]
        if target_tile.is_purchasable and target_tile.owner is not None and target_tile.owner != current_player_id and not target_tile.is_mortgaged:
            r = board.calculate_rent(target_pos, roll)
            if r > max_threat:
                max_threat = float(r)
    obs.append(float(np.tanh(max_threat / 500.0)))

    return np.array(obs, dtype=np.float32)

