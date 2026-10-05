"""
Game Replay Recorder.
Generates structured JSON telemetry of every turn, move, cash transfer, and board state
for interactive web visualization.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from typing import Any
from monopoly_core import MonopolyGame
from agents import RLAgent, MarkovROIAgent, AggressiveAgent, ConservativeAgent


def record_game(seed: int = 5000, max_turns: int = 250, output_file: str = "visualizer/replay.json") -> dict[str, Any]:
    game = MonopolyGame(seed=seed, max_turns=max_turns, enable_logging=True)
    game.players[0].name = "PPO-Agent (Trained RL)"
    game.players[0].agent = RLAgent()
    game.players[1].name = "Markov-ROI Bot"
    game.players[1].agent = MarkovROIAgent()
    game.players[2].name = "Aggressive Bot"
    game.players[2].agent = AggressiveAgent()
    game.players[3].name = "Conservative Bot"
    game.players[3].agent = ConservativeAgent()

    frames = []

    def capture_frame(event_type: str, details: str, dice: tuple = (0, 0)):
        tiles_state = []
        for t in game.board.tiles:
            tiles_state.append({
                "index": t.index,
                "name": t.name,
                "owner": t.owner,
                "num_houses": t.num_houses,
                "num_hotels": t.num_hotels,
                "is_mortgaged": t.is_mortgaged,
            })

        players_state = []
        for p in game.players:
            players_state.append({
                "id": p.player_id,
                "name": p.name,
                "cash": p.cash,
                "position": p.position,
                "in_jail": p.in_jail,
                "is_bankrupt": p.is_bankrupt,
                "net_worth": p.net_worth(game.board),
            })

        frames.append({
            "turn": game.current_turn,
            "active_player": game.current_player_idx,
            "event_type": event_type,
            "details": details,
            "dice": list(dice),
            "available_houses": game.board.available_houses,
            "available_hotels": game.board.available_hotels,
            "players": players_state,
            "tiles": tiles_state,
        })

    capture_frame("START", "Game started with trained PPO-Agent vs Markov-ROI League.")

    while not game.game_over:
        curr_p = game.players[game.current_player_idx]
        if curr_p.is_bankrupt:
            game._advance_turn()
            continue

        turn_start_cash = curr_p.cash
        pos_before = curr_p.position

        # Execute turn
        active = game.execute_turn()
        pos_after = curr_p.position
        cash_delta = curr_p.cash - turn_start_cash

        last_log = game.log_messages[-1] if game.log_messages else f"{curr_p.name} moved."
        capture_frame("MOVE", last_log)

    winner_name = game.players[game.winner_id].name if game.winner_id is not None else "Draw"
    capture_frame("GAME_OVER", f"Game Over! Winner: {winner_name}")

    replay_data = {
        "seed": seed,
        "max_turns": max_turns,
        "total_frames": len(frames),
        "winner_id": game.winner_id,
        "winner_name": winner_name,
        "frames": frames,
    }

    with open(output_file, "w") as f:
        json.dump(replay_data, f, indent=2)

    js_file = output_file.replace(".json", "_data.js")
    with open(js_file, "w") as f:
        f.write(f"window.REPLAY_DATA = {json.dumps(replay_data, indent=2)};\n")

    print(f"Recorded {len(frames)} replay frames to '{output_file}' and '{js_file}'. Winner: {winner_name}")
    return replay_data


if __name__ == "__main__":
    # Find a victory seed for PPO-Agent
    print("Finding a showcase victory match for trained PPO-Agent...")
    for seed in range(5000, 5050):
        g = MonopolyGame(seed=seed, max_turns=250)
        g.players[0].agent = RLAgent()
        g.players[1].agent = MarkovROIAgent()
        g.players[2].agent = AggressiveAgent()
        g.players[3].agent = ConservativeAgent()
        while not g.game_over:
            g.execute_turn()
        if g.winner_id == 0:
            print(f"Found winning seed {seed}! Generating full telemetry replay...")
            record_game(seed=seed)
            break
