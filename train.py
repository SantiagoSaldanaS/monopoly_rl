"""
Masked PPO Training Pipeline for Monopoly on PyTorch + CUDA.
Trains an RL agent against the League (Markov-ROI, Aggressive, Conservative).
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from monopoly_env import MonopolyEnv, Action, TOTAL_ACTIONS
from models import MaskedActorCritic
from agents import AggressiveAgent, ConservativeAgent, RandomAgent, MarkovROIAgent


class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.masks = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        self.obs.clear()
        self.masks.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()


def train_ppo(
    total_episodes: int = 500,
    rollout_steps: int = 1000,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    epochs_per_update: int = 4,
    batch_size: int = 128,
    save_interval: int = 50,
    resume_path: str = None,
    league: str = "grandmaster",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    from agents.markov_agent import MarkovROIAgent
    from agents.baseline_agents import AggressiveAgent, ConservativeAgent
    from agents.grandmaster_agent import TournamentGrandmasterAgent

    if league == "grandmaster":
        league_name = "Grandmaster League (GrandmasterBot, Markov-ROI, Aggressive)"
        opponents = [TournamentGrandmasterAgent(), MarkovROIAgent(), AggressiveAgent()]
    elif league == "elite":
        league_name = "Elite Shark Tank (2x GrandmasterBot, Markov-ROI)"
        opponents = [TournamentGrandmasterAgent(), TournamentGrandmasterAgent(), MarkovROIAgent()]
    elif league == "all-markov":
        league_name = "3x Markov-ROI"
        opponents = [MarkovROIAgent(), MarkovROIAgent(), MarkovROIAgent()]
    else:
        league_name = "Mixed (Markov-ROI, Aggressive, Conservative)"
        opponents = [MarkovROIAgent(), AggressiveAgent(), ConservativeAgent()]

    print(f"\n=======================================================")
    print(f"Starting Monopoly Masked-PPO League Training")
    print(f"Device: {device.upper()} (GPU: {torch.cuda.get_device_name(0) if device == 'cuda' else 'N/A'})")
    print(f"Total Episodes: {total_episodes} | Steps per update: {rollout_steps}")
    print(f"Opponent League: {league_name}")
    print(f"=======================================================\n")

    os.makedirs("checkpoints", exist_ok=True)

    # Initialize model on GPU
    model = MaskedActorCritic(obs_dim=439, act_dim=TOTAL_ACTIONS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if resume_path and os.path.exists(resume_path):
        print(f"[INFO] Resuming training from checkpoint: '{resume_path}'")
        try:
            model.load_state_dict(torch.load(resume_path, map_location=device))
        except Exception as e:
            print(f"[WARN] Could not load checkpoint ({e}), initializing fresh model.")

    buffer = RolloutBuffer()
    env = MonopolyEnv(max_turns=300, opponents=opponents)
    obs_dict, info_dict = env.reset()

    # Track metrics
    start_time = time.time()
    total_env_steps = 0
    rl_agent_wins = 0
    episodes_completed = 0
    window_wins = 0
    window_episodes = 0

    for episode in range(1, total_episodes + 1):
        step_count = 0

        while step_count < rollout_steps:
            obs_raw = obs_dict["player_0"]
            mask_raw = info_dict["player_0"]["action_mask"]

            obs_tensor = torch.tensor(obs_raw, dtype=torch.float32, device=device).unsqueeze(0)
            mask_tensor = torch.tensor(mask_raw, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                dist, value = model(obs_tensor, mask_tensor)
                action_tensor = dist.sample()
                log_prob = dist.log_prob(action_tensor)

            action = action_tensor.item()

            # Step environment (runs Player 0 action, then opponents automatically respond)
            next_obs, rewards, term, trunc, next_info = env.step(action)
            total_env_steps += 1
            step_count += 1

            reward = rewards["player_0"]
            done = term["player_0"]

            buffer.obs.append(obs_raw)
            buffer.masks.append(mask_raw)
            buffer.actions.append(action)
            buffer.log_probs.append(log_prob.item())
            buffer.values.append(value.item())
            buffer.rewards.append(reward)
            buffer.dones.append(done)

            obs_dict = next_obs
            info_dict = next_info

            # Check game over or bankruptcy
            if done:
                episodes_completed += 1
                window_episodes += 1
                if env.game.winner_id == 0:
                    rl_agent_wins += 1
                    window_wins += 1

                obs_dict, info_dict = env.reset()

        # Update PPO policy after collecting rollout buffer
        if len(buffer.obs) > 0:
            obs_b = torch.tensor(np.array(buffer.obs), dtype=torch.float32, device=device)
            mask_b = torch.tensor(np.array(buffer.masks), dtype=torch.float32, device=device)
            act_b = torch.tensor(np.array(buffer.actions), dtype=torch.int64, device=device)
            old_log_probs_b = torch.tensor(np.array(buffer.log_probs), dtype=torch.float32, device=device)
            rewards_b = np.array(buffer.rewards, dtype=np.float32)
            dones_b = np.array(buffer.dones, dtype=np.float32)
            values_b = np.array(buffer.values, dtype=np.float32)

            # Compute Generalized Advantage Estimation (GAE) with critic bootstrapping
            if not dones_b[-1]:
                with torch.no_grad():
                    obs_last = torch.tensor(obs_dict["player_0"], dtype=torch.float32, device=device).unsqueeze(0)
                    mask_last = torch.tensor(info_dict["player_0"]["action_mask"], dtype=torch.float32, device=device).unsqueeze(0)
                    _, next_val_last = model(obs_last, mask_last)
                    bootstrap_val = next_val_last.item()
            else:
                bootstrap_val = 0.0

            advantages = np.zeros_like(rewards_b, dtype=np.float32)
            last_gae = 0.0
            for t in reversed(range(len(rewards_b))):
                next_val = values_b[t + 1] if t + 1 < len(rewards_b) else bootstrap_val
                delta = rewards_b[t] + gamma * next_val * (1.0 - dones_b[t]) - values_b[t]
                last_gae = delta + gamma * gae_lambda * (1.0 - dones_b[t]) * last_gae
                advantages[t] = last_gae

            returns = advantages + values_b
            adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
            returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)

            # Normalize advantages
            adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

            # PPO SGD Updates
            dataset_size = len(buffer.obs)
            indices = np.arange(dataset_size)

            for _ in range(epochs_per_update):
                np.random.shuffle(indices)
                for start in range(0, dataset_size, batch_size):
                    batch_idx = indices[start : start + batch_size]

                    batch_obs = obs_b[batch_idx]
                    batch_masks = mask_b[batch_idx]
                    batch_act = act_b[batch_idx]
                    batch_old_lp = old_log_probs_b[batch_idx]
                    batch_adv = adv_tensor[batch_idx]
                    batch_ret = returns_tensor[batch_idx]

                    dist, new_values = model(batch_obs, batch_masks)
                    new_log_probs = dist.log_prob(batch_act)

                    # Policy ratio
                    ratio = torch.exp(new_log_probs - batch_old_lp)
                    surr1 = ratio * batch_adv
                    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * batch_adv

                    # Choice mask: only update policy on steps where agent had a real choice (more than 1 legal action)
                    choice_mask = (batch_masks.sum(dim=-1) > 1.0).float()
                    if choice_mask.sum() > 0:
                        policy_loss = -(torch.min(surr1, surr2) * choice_mask).sum() / choice_mask.sum()
                        entropy = (dist.entropy() * choice_mask).sum() / choice_mask.sum()
                    else:
                        policy_loss = torch.tensor(0.0, device=device)
                        entropy = dist.entropy().mean()

                    # Value loss (computed across ALL states)
                    value_loss = nn.functional.mse_loss(new_values.squeeze(-1), batch_ret)

                    # Total loss (balanced entropy)
                    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    optimizer.step()

            buffer.clear()

        # Log progress every 20 episodes
        if episode % 20 == 0 or episode == total_episodes:
            elapsed = time.time() - start_time
            overall_wr = (rl_agent_wins / max(1, episodes_completed)) * 100
            recent_wr = (window_wins / max(1, window_episodes)) * 100
            sps = total_env_steps / elapsed
            print(
                f"Episode {episode:4d}/{total_episodes} | Steps: {total_env_steps:7d} "
                f"| Recent Win Rate: {recent_wr:5.1f}% ({window_wins}/{window_episodes}) "
                f"| Overall: {overall_wr:5.1f}% "
                f"| Speed: {sps:6.1f} steps/s"
            )
            window_wins = 0
            window_episodes = 0

        # Periodic checkpointing
        if episode % save_interval == 0:
            ckpt_path = f"checkpoints/monopoly_ppo_ep{episode}.pt"
            torch.save(model.state_dict(), ckpt_path)
            torch.save(model.state_dict(), "checkpoints/monopoly_ppo_final.pt")

    # Save final model
    torch.save(model.state_dict(), "checkpoints/monopoly_ppo_final.pt")
    print(f"\n[SUCCESS] Model successfully trained and saved to 'checkpoints/monopoly_ppo_final.pt'!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Monopoly PPO Agent")
    parser.add_argument("--episodes", "-e", type=int, default=1000, help="Total training episodes (default: 1000)")
    parser.add_argument("--rollout-steps", "-s", type=int, default=1000, help="Steps per PPO update (default: 1000)")
    parser.add_argument("--save-interval", type=int, default=100, help="Checkpoint save interval (default: 100)")
    parser.add_argument("--resume", "-r", type=str, default="checkpoints/monopoly_ppo_final.pt", help="Path to checkpoint to resume from")
    parser.add_argument("--scratch", action="store_true", help="Start fresh without loading existing checkpoint")
    parser.add_argument(
        "--league",
        "-l",
        type=str,
        default="grandmaster",
        choices=["grandmaster", "elite", "all-markov", "mixed"],
        help="Opponent League: 'grandmaster' (46% GrandmasterBot + Markov + Aggressive), 'elite' (2x Grandmaster + Markov), 'all-markov' (3x Markov), 'mixed'",
    )
    args = parser.parse_args()

    resume_file = None if args.scratch else args.resume
    train_ppo(
        total_episodes=args.episodes,
        rollout_steps=args.rollout_steps,
        save_interval=args.save_interval,
        resume_path=resume_file,
        league=args.league,
    )
