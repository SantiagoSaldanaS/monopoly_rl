"""
Self-Play PPO Training Pipeline for Monopoly.
All 4 seats are played by the neural network, generating 4x data throughput
and natural curriculum learning (AlphaZero style).
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
from tournament import run_tournament, AggressiveAgent, ConservativeAgent, RandomAgent
from agents import MarkovROIAgent


class SelfPlayBuffer:
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


def train_self_play(
    total_episodes: int = 500,
    rollout_steps: int = 2000,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    epochs_per_update: int = 4,
    batch_size: int = 256,
    eval_interval: int = 50,
    save_interval: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print(f"\n=======================================================")
    print(f"Starting Monopoly Self-Play PPO Training (AlphaZero Style)")
    print(f"Device: {device.upper()} (GPU: {torch.cuda.get_device_name(0) if device == 'cuda' else 'N/A'})")
    print(f"Total Episodes: {total_episodes} | Steps per update: {rollout_steps}")
    print(f"=======================================================\n")

    os.makedirs("checkpoints", exist_ok=True)

    model = MaskedActorCritic(obs_dim=304, act_dim=TOTAL_ACTIONS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    buffer = SelfPlayBuffer()
    env = MonopolyEnv(max_turns=300)
    # Clear heuristic agents so all players are neural network driven
    for p in env.game.players:
        p.agent = None

    obs_dict, info_dict = env.reset()

    start_time = time.time()
    total_steps = 0
    games_completed = 0

    for episode in range(1, total_episodes + 1):
        step_count = 0

        while step_count < rollout_steps:
            curr_id = env.game.current_player_idx
            agent_key = f"player_{curr_id}"

            obs_raw = obs_dict[agent_key]
            mask_raw = info_dict[agent_key]["action_mask"]

            obs_tensor = torch.tensor(obs_raw, dtype=torch.float32, device=device).unsqueeze(0)
            mask_tensor = torch.tensor(mask_raw, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                dist, value = model(obs_tensor, mask_tensor)
                action_tensor = dist.sample()
                log_prob = dist.log_prob(action_tensor)

            action = action_tensor.item()

            next_obs, rewards, term, trunc, next_info = env.step(action)
            total_steps += 1
            step_count += 1

            reward = rewards[agent_key]
            done = term[agent_key]

            buffer.obs.append(obs_raw)
            buffer.masks.append(mask_raw)
            buffer.actions.append(action)
            buffer.log_probs.append(log_prob.item())
            buffer.values.append(value.item())
            buffer.rewards.append(reward)
            buffer.dones.append(done)

            obs_dict = next_obs
            info_dict = next_info

            if any(term.values()):
                games_completed += 1
                obs_dict, info_dict = env.reset()
                for p in env.game.players:
                    p.agent = None

        # PPO Update
        if len(buffer.obs) > 0:
            obs_b = torch.tensor(np.array(buffer.obs), dtype=torch.float32, device=device)
            mask_b = torch.tensor(np.array(buffer.masks), dtype=torch.float32, device=device)
            act_b = torch.tensor(np.array(buffer.actions), dtype=torch.int64, device=device)
            old_log_probs_b = torch.tensor(np.array(buffer.log_probs), dtype=torch.float32, device=device)
            rewards_b = np.array(buffer.rewards, dtype=np.float32)
            dones_b = np.array(buffer.dones, dtype=np.float32)
            values_b = np.array(buffer.values, dtype=np.float32)

            advantages = np.zeros_like(rewards_b, dtype=np.float32)
            last_gae = 0.0
            for t in reversed(range(len(rewards_b))):
                next_val = values_b[t + 1] if t + 1 < len(rewards_b) and not dones_b[t] else 0.0
                delta = rewards_b[t] + gamma * next_val * (1.0 - dones_b[t]) - values_b[t]
                last_gae = delta + gamma * gae_lambda * (1.0 - dones_b[t]) * last_gae
                advantages[t] = last_gae

            returns = advantages + values_b
            adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
            returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
            adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

            dataset_size = len(buffer.obs)
            indices = np.arange(dataset_size)

            for _ in range(epochs_per_update):
                np.random.shuffle(indices)
                for start in range(0, dataset_size, batch_size):
                    batch_idx = indices[start : start + batch_size]

                    dist, new_values = model(obs_b[batch_idx], mask_b[batch_idx])
                    new_log_probs = dist.log_prob(act_b[batch_idx])
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_log_probs - old_log_probs_b[batch_idx])
                    surr1 = ratio * adv_tensor[batch_idx]
                    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_tensor[batch_idx]
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = nn.functional.mse_loss(new_values.squeeze(-1), returns_tensor[batch_idx])
                    loss = policy_loss + 0.5 * value_loss - 0.005 * entropy

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    optimizer.step()

            buffer.clear()

        # Log throughput
        if episode % 10 == 0 or episode == total_episodes:
            elapsed = time.time() - start_time
            sps = total_steps / elapsed
            print(f"Episode {episode:4d}/{total_episodes} | Steps: {total_steps:7d} | Games Completed: {games_completed:5d} | Speed: {sps:6.1f} steps/s")

        if episode % save_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/selfplay_ep{episode}.pt")
            torch.save(model.state_dict(), "checkpoints/selfplay_final.pt")

    torch.save(model.state_dict(), "checkpoints/selfplay_final.pt")
    print(f"\n[SUCCESS] Self-play model saved to 'checkpoints/selfplay_final.pt'!")


if __name__ == "__main__":
    train_self_play(total_episodes=500, rollout_steps=2000)
