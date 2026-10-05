"""
Tournament Benchmark Evaluator for RL Agent.
Plays multi-game tournaments against the Grandmaster League with seat rotation.
"""

import argparse
import time
from collections import defaultdict
import numpy as np

from monopoly_core.game import MonopolyGame
from agents.rl_agent import RLAgent
from agents.grandmaster_agent import TournamentGrandmasterAgent
from agents.markov_agent import MarkovROIAgent
from agents.baseline_agents import AggressiveAgent, ConservativeAgent


def evaluate_tournament(
    model_path: str = "checkpoints/monopoly_ppo_final.pt",
    num_games: int = 100,
    max_turns: int = 350,
    device: str = "cuda",
    league: str = "grandmaster",
):
    rl_agent = RLAgent(model_path=model_path, device=device, name="RL-Agent (PPO)")
    if league == "arena":
        league_title = "Active Arena Bots (GrandmasterBot, Markov-ROI, Conservative)"
        competitors = [
            ("RL-Agent (PPO)", rl_agent),
            ("GrandmasterBot", TournamentGrandmasterAgent(name="GrandmasterBot")),
            ("Markov-ROI", MarkovROIAgent(name="Markov-ROI")),
            ("Conservative", ConservativeAgent(name="Conservative")),
        ]
    elif league == "mixed":
        league_title = "Mixed League (Markov-ROI, Aggressive, Conservative)"
        competitors = [
            ("RL-Agent (PPO)", rl_agent),
            ("Markov-ROI", MarkovROIAgent(name="Markov-ROI")),
            ("Aggressive", AggressiveAgent(name="Aggressive")),
            ("Conservative", ConservativeAgent(name="Conservative")),
        ]
    else:
        league_title = "Grandmaster League (GrandmasterBot, Markov-ROI, Aggressive)"
        competitors = [
            ("RL-Agent (PPO)", rl_agent),
            ("GrandmasterBot", TournamentGrandmasterAgent(name="GrandmasterBot")),
            ("Markov-ROI", MarkovROIAgent(name="Markov-ROI")),
            ("Aggressive", AggressiveAgent(name="Aggressive")),
        ]

    print(f"\n=======================================================")
    print(f"TOURNAMENT BENCHMARK: RL Agent vs {league_title}")
    print(f"Model: {model_path} | Device: {device.upper()}")
    print(f"Games: {num_games} | Max Turns: {max_turns}")
    print(f"=======================================================\n")

    wins = defaultdict(int)
    bankruptcies = defaultdict(int)
    turns_list = []
    t0 = time.time()

    for game_idx in range(num_games):
        game = MonopolyGame(num_players=4, seed=3000 + game_idx, max_turns=max_turns)

        # Rotate seats each game for strict fairness
        shift = game_idx % 4
        seat_mapping = {}
        for seat in range(4):
            comp_idx = (seat + shift) % 4
            name, agent = competitors[comp_idx]
            game.players[seat].name = name
            game.players[seat].agent = agent
            seat_mapping[seat] = name

        winner_id = game.play_full_game()
        turns_list.append(game.current_turn)

        for p in game.players:
            if p.is_bankrupt:
                bankruptcies[p.name] += 1

        if winner_id is not None:
            winner_name = game.players[winner_id].name
            wins[winner_name] += 1
        else:
            wins["Draw"] += 1

        if (game_idx + 1) % 25 == 0 or (game_idx + 1) == num_games:
            rl_w = wins["RL-Agent (PPO)"]
            print(f"Game {game_idx + 1:3d}/{num_games} | RL Wins: {rl_w} ({rl_w / (game_idx + 1) * 100:.1f}%)")

    elapsed = time.time() - t0
    speed = num_games / elapsed

    print(f"\n=======================================================")
    print(f"FINAL TOURNAMENT RESULTS ({num_games} Games)")
    print(f"Elapsed Time: {elapsed:.2f}s ({speed:.1f} games/sec)")
    print(f"Average Game Length: {np.mean(turns_list):.1f} turns")
    print(f"-------------------------------------------------------")
    for name, _ in competitors:
        w = wins[name]
        pct = (w / num_games) * 100
        b = bankruptcies[name]
        print(f"  {name:18s}: {w:3d} wins ({pct:5.1f}%) | Bankrupt: {b:3d} times")
    if wins["Draw"] > 0:
        print(f"  {'Draws':18s}: {wins['Draw']:3d} ({wins['Draw'] / num_games * 100:5.1f}%)")
    print(f"=======================================================\n")

    return wins


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Monopoly RL Agent")
    parser.add_argument("--model", "-m", type=str, default="checkpoints/monopoly_ppo_final.pt")
    parser.add_argument("--games", "-g", type=int, default=100)
    parser.add_argument("--max-turns", "-t", type=int, default=350)
    parser.add_argument("--device", "-d", type=str, default="cuda")
    parser.add_argument("--league", "-l", type=str, default="arena", choices=["grandmaster", "mixed", "arena"])
    args = parser.parse_args()

    evaluate_tournament(
        model_path=args.model,
        num_games=args.games,
        max_turns=args.max_turns,
        device=args.device,
        league=args.league,
    )
