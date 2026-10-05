"""
Tournament and Evaluation Harness for Monopoly Agents.
Simulates matches across different strategic playstyles and outputs win-rate statistics.
"""

import time
from collections import defaultdict
from typing import Any
import numpy as np

from monopoly_env import MonopolyEnv
from agents import RandomAgent, AggressiveAgent, ConservativeAgent


def run_tournament(
    agent_classes: list,
    num_games: int = 100,
    max_turns: int = 350,
) -> dict[str, Any]:
    """
    Runs a round-robin / multi-game tournament among 4 agent types.
    """
    win_counts = defaultdict(int)
    total_turns = []
    t0 = time.time()

    for game_idx in range(num_games):
        env = MonopolyEnv(max_turns=max_turns, seed=2000 + game_idx)
        obs, info = env.reset()

        # Instantiate agents for this game
        agents = [cls(name=f"{cls.__name__}_{i}") for i, cls in enumerate(agent_classes)]

        done = False
        while not done:
            curr_id = env.current_agent_idx
            agent_key = f"player_{curr_id}"
            agent = agents[curr_id]

            mask = info[agent_key]["action_mask"]
            agent_obs = obs[agent_key]

            action = agent.select_action(agent_obs, mask)
            obs, rewards, term, trunc, info = env.step(action)

            if any(term.values()):
                done = True
                winner_id = env.game.winner_id
                if winner_id is not None:
                    winner_name = agent_classes[winner_id].__name__
                    win_counts[winner_name] += 1
                else:
                    win_counts["Draw"] += 1
                total_turns.append(env.game.current_turn)

    elapsed = time.time() - t0
    speed = num_games / elapsed

    print(f"\n=======================================================")
    print(f"TOURNAMENT RESULTS ({num_games} Games)")
    print(f"Elapsed Time: {elapsed:.2f}s ({speed:.1f} games/sec)")
    print(f"Average Game Length: {np.mean(total_turns):.1f} turns")
    print(f"-------------------------------------------------------")
    for agent_name, wins in sorted(win_counts.items(), key=lambda x: x[1], reverse=True):
        win_rate = (wins / num_games) * 100
        print(f"  {agent_name:20s}: {wins:4d} wins ({win_rate:5.1f}%)")
    print(f"=======================================================\n")

    return dict(win_counts)


if __name__ == "__main__":
    # Test match: 2 Aggressive vs 1 Conservative vs 1 Random
    lineup = [AggressiveAgent, AggressiveAgent, ConservativeAgent, RandomAgent]
    run_tournament(lineup, num_games=100)
