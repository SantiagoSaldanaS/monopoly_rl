"""
Tournament PPO Training Pipeline for Monopoly on PyTorch + CUDA.
Designed for 12+ hour overnight training to achieve peak dominance over
TournamentGrandmasterAgent, MarkovROIAgent, and AggressiveAgent.
"""

import os
import sys
import time
import argparse
import random
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from monopoly_env import MonopolyEnv, Action, TOTAL_ACTIONS
from monopoly_core.constants import COLOR_GROUP_TILES, ColorGroup, JAIL_FINE
from monopoly_env.action_space import get_action_mask, BUILD_GROUP_START, BUILDABLE_GROUPS
from monopoly_env.obs_encoder import encode_observation
from monopoly_core.game import MonopolyGame
from models import MaskedActorCritic
from agents import AggressiveAgent, ConservativeAgent, RandomAgent, MarkovROIAgent
from agents.grandmaster_agent import TournamentGrandmasterAgent
from agents.rl_agent import RLAgent


class DualLogger:
    def __init__(self, filename="training_overnight.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


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


def warmup_expert_demonstrations(model: MaskedActorCritic, device: str, num_games: int = 400):
    """
    Rapid behavioral cloning pre-training on expert games.
    Ensures network starts with full knowledge of deed acquisition, building, and cash flow.
    """
    print(f"\n[WARMUP] Collecting {num_games} expert games for behavioral cloning pre-training...", flush=True)
    t0 = time.time()
    gm = TournamentGrandmasterAgent()
    agg = AggressiveAgent()
    markov = MarkovROIAgent()

    def get_expert_action(game, pid, pending, mask):
        p = game.players[pid]
        b = game.board
        if pending == "BUY_OR_AUCTION":
            t = b.tiles[p.position]
            if (gm.on_buy_decision(p, t, game) or p.cash >= t.price + 100) and mask[Action.BUY_PROPERTY] == 1:
                return Action.BUY_PROPERTY
            return Action.DECLINE_TO_AUCTION
        if pending == "JAIL":
            unowned = sum(1 for t in b.tiles if t.is_purchasable and t.owner is None)
            if unowned > 6:
                if p.get_out_of_jail_cards > 0 and mask[Action.USE_JAIL_CARD] == 1:
                    return Action.USE_JAIL_CARD
                if p.cash >= JAIL_FINE + 150 and mask[Action.PAY_JAIL_FINE] == 1:
                    return Action.PAY_JAIL_FINE
            return Action.END_OR_ROLL
        reserve = gm._get_dynamic_cash_reserve(p, game)
        for g in [ColorGroup.ORANGE, ColorGroup.RED, ColorGroup.SKY_BLUE, ColorGroup.YELLOW, ColorGroup.PINK, ColorGroup.DARK_BLUE, ColorGroup.GREEN, ColorGroup.BROWN]:
            if b.owns_full_group(pid, g):
                g_idx = BUILDABLE_GROUPS.index(g)
                act = BUILD_GROUP_START + g_idx
                if mask[act] == 1:
                    indices = COLOR_GROUP_TILES[g]
                    min_h = min(b.tiles[i].num_houses + (5 if b.tiles[i].num_hotels else 0) for i in indices)
                    if min_h == 4 and b.available_houses <= 10:
                        continue
                    cost = b.tiles[indices[0]].house_cost
                    if p.cash >= cost + reserve:
                        return act
        return Action.END_OR_ROLL

    obs_list, mask_list, act_list, val_list = [], [], [], []
    for g in range(num_games):
        game = MonopolyGame(num_players=4, seed=10000 + g, max_turns=250)
        # Alternate expert seat between 0 and other players
        game.players[0].agent = gm
        game.players[1].agent = markov
        game.players[2].agent = agg
        game.players[3].agent = gm

        game_obs, game_masks, game_acts = [], [], []
        while not game.game_over:
            p0 = game.players[0]
            if not p0.is_bankrupt and game.current_player_idx == 0:
                obs = encode_observation(game, 0)
                mask = get_action_mask(game, 0, "MANAGEMENT")
                act = get_expert_action(game, 0, "MANAGEMENT", mask)
                game_obs.append(obs)
                game_masks.append(mask)
                game_acts.append(act)

            p_curr = game.players[game.current_player_idx]
            if game.current_player_idx == 0 and not p_curr.in_jail:
                tile = game.board.tiles[p_curr.position]
                if tile.is_purchasable and tile.owner is None:
                    b_obs = encode_observation(game, 0)
                    b_mask = get_action_mask(game, 0, "BUY_OR_AUCTION")
                    b_act = get_expert_action(game, 0, "BUY_OR_AUCTION", b_mask)
                    game_obs.append(b_obs)
                    game_masks.append(b_mask)
                    game_acts.append(b_act)

            game.execute_turn()

        target_val = 1.0 if game.winner_id == 0 else (-1.0 if game.players[0].is_bankrupt else 0.0)
        obs_list.extend(game_obs)
        mask_list.extend(game_masks)
        act_list.extend(game_acts)
        val_list.extend([target_val] * len(game_obs))

    dataset_size = len(obs_list)
    print(f"[WARMUP] Collected {dataset_size} expert transitions in {time.time() - t0:.1f}s. Training actor-critic...", flush=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion_act = nn.CrossEntropyLoss()
    criterion_val = nn.MSELoss()

    obs_t = torch.tensor(np.array(obs_list), dtype=torch.float32, device=device)
    mask_t = torch.tensor(np.array(mask_list), dtype=torch.float32, device=device)
    act_t = torch.tensor(np.array(act_list), dtype=torch.int64, device=device)
    val_t = torch.tensor(np.array(val_list), dtype=torch.float32, device=device).unsqueeze(1)

    batch_size = 256
    indices = np.arange(dataset_size)

    for epoch in range(8):
        np.random.shuffle(indices)
        for start in range(0, dataset_size, batch_size):
            idx = indices[start : start + batch_size]
            dist, val = model(obs_t[idx], mask_t[idx])
            loss = criterion_act(dist.logits, act_t[idx]) + 0.5 * criterion_val(val, val_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(f"[WARMUP] Completed behavioral cloning warmup in {time.time() - t0:.1f}s!\n", flush=True)


def evaluate_tournament(model: MaskedActorCritic, device: str, num_games: int = 30) -> dict[str, float]:
    """
    Evaluates policy in head-to-head tournament against Grandmaster League with strict seat rotation.
    """
    model.eval()
    temp_ckpt = "checkpoints/_eval_temp.pt"
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), temp_ckpt)

    rl_agent = RLAgent(model_path=temp_ckpt, device=device, name="RL-Agent (PPO)")
    gm_agent = TournamentGrandmasterAgent(name="GrandmasterBot")
    markov_agent = MarkovROIAgent(name="Markov-ROI")
    agg_agent = AggressiveAgent(name="Aggressive")

    competitors = [
        ("RL-Agent (PPO)", rl_agent),
        ("GrandmasterBot", gm_agent),
        ("Markov-ROI", markov_agent),
        ("Aggressive", agg_agent),
    ]

    wins = defaultdict(int)
    bankruptcies = defaultdict(int)

    for g_idx in range(num_games):
        game = MonopolyGame(num_players=4, seed=4000 + g_idx, max_turns=350)
        shift = g_idx % 4
        for seat in range(4):
            comp_idx = (seat + shift) % 4
            name, agent = competitors[comp_idx]
            game.players[seat].name = name
            game.players[seat].agent = agent

        winner_id = game.play_full_game()
        for p in game.players:
            if p.is_bankrupt:
                bankruptcies[p.name] += 1
        if winner_id is not None:
            wins[game.players[winner_id].name] += 1
        else:
            wins["Draw"] += 1

    try:
        os.remove(temp_ckpt)
    except OSError:
        pass

    model.train()
    results = {name: (wins[name] / num_games) * 100 for name, _ in competitors}
    results["Draws"] = (wins["Draw"] / num_games) * 100
    return results


def train_ppo(
    total_episodes: int = 10000,
    target_hours: float = 12.5,
    rollout_steps: int = 2048,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    epochs_per_update: int = 4,
    batch_size: int = 256,
    save_interval: int = 100,
    eval_interval: int = 100,
    resume_path: str = None,
    warmup: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print("=" * 65, flush=True)
    print("MONOPOLY REINFORCEMENT LEARNING OVERNIGHT PPO TRAINER", flush=True)
    print(f"Device: {device.upper()} (GPU: {torch.cuda.get_device_name(0) if device == 'cuda' else 'N/A'})", flush=True)
    print(f"Target Duration: {target_hours:.1f} Hours | Max Updates: {total_episodes}", flush=True)
    print(f"Rollout Steps: {rollout_steps} | Batch Size: {batch_size} | Epochs/Update: {epochs_per_update}", flush=True)
    print("=" * 65 + "\n", flush=True)

    os.makedirs("checkpoints", exist_ok=True)

    model = MaskedActorCritic(obs_dim=439, act_dim=TOTAL_ACTIONS).to(device)

    # Load or warm-start
    if resume_path and os.path.exists(resume_path):
        print(f"[INFO] Resuming training from checkpoint: '{resume_path}'", flush=True)
        try:
            model.load_state_dict(torch.load(resume_path, map_location=device))
        except Exception as e:
            print(f"[WARN] Could not load checkpoint ({e}), initializing fresh model.", flush=True)
            if warmup:
                warmup_expert_demonstrations(model, device)
    elif warmup:
        warmup_expert_demonstrations(model, device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Cosine annealing scheduler over target duration
    # Estimate total updates in target_hours (approx 5.5s per iteration on RTX 4090)
    estimated_total_updates = int((target_hours * 3600) / 5.5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(total_episodes, estimated_total_updates),
        eta_min=3e-5,
    )

    buffer = RolloutBuffer()

    # Dynamic opponent pool for broad strategic dominance
    opponent_pool = [
        TournamentGrandmasterAgent(),
        MarkovROIAgent(),
        AggressiveAgent(),
        ConservativeAgent(),
    ]

    def sample_league():
        # Grandmaster + Markov + Aggressive is the core competitive benchmark
        if random.random() < 0.65:
            return [TournamentGrandmasterAgent(), MarkovROIAgent(), AggressiveAgent()]
        return random.sample(opponent_pool, 3)

    env = MonopolyEnv(max_turns=350, opponents=sample_league())
    obs_dict, info_dict = env.reset()

    # Metrics
    start_time = time.time()
    target_seconds = target_hours * 3600
    total_env_steps = 0
    rl_agent_wins = 0
    episodes_completed = 0
    window_wins = 0
    window_episodes = 0
    best_eval_win_rate = 0.0

    print("[INFO] Starting PPO training loop...\n", flush=True)

    iteration = 0
    while True:
        iteration += 1
        elapsed_time = time.time() - start_time

        # Check termination condition: target hours reached or max updates reached
        if elapsed_time >= target_seconds:
            print(f"\n[TARGET REACHED] Target duration of {target_hours:.1f} hours elapsed! ({elapsed_time/3600:.2f} hrs)", flush=True)
            break
        if iteration > total_episodes:
            print(f"\n[TARGET REACHED] Maximum episode update count of {total_episodes} reached!", flush=True)
            break

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

            if done:
                episodes_completed += 1
                window_episodes += 1
                if env.game.winner_id == 0:
                    rl_agent_wins += 1
                    window_wins += 1

                # Sample diverse opponents on game reset
                env.default_opponents = sample_league()
                obs_dict, info_dict = env.reset()

        # PPO Policy Update
        if len(buffer.obs) > 0:
            obs_b = torch.tensor(np.array(buffer.obs), dtype=torch.float32, device=device)
            mask_b = torch.tensor(np.array(buffer.masks), dtype=torch.float32, device=device)
            act_b = torch.tensor(np.array(buffer.actions), dtype=torch.int64, device=device)
            old_log_probs_b = torch.tensor(np.array(buffer.log_probs), dtype=torch.float32, device=device)
            rewards_b = np.array(buffer.rewards, dtype=np.float32)
            dones_b = np.array(buffer.dones, dtype=np.float32)
            values_b = np.array(buffer.values, dtype=np.float32)

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
            adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

            dataset_size = len(buffer.obs)
            indices = np.arange(dataset_size)

            for _ in range(epochs_per_update):
                np.random.shuffle(indices)
                for start in range(0, dataset_size, batch_size):
                    batch_idx = indices[start : start + batch_size]

                    b_obs = obs_b[batch_idx]
                    b_masks = mask_b[batch_idx]
                    b_act = act_b[batch_idx]
                    b_old_lp = old_log_probs_b[batch_idx]
                    b_adv = adv_tensor[batch_idx]
                    b_ret = returns_tensor[batch_idx]

                    dist, new_values = model(b_obs, b_masks)
                    new_log_probs = dist.log_prob(b_act)

                    ratio = torch.exp(new_log_probs - b_old_lp)
                    surr1 = ratio * b_adv
                    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv

                    choice_mask = (b_masks.sum(dim=-1) > 1.0).float()
                    if choice_mask.sum() > 0:
                        policy_loss = -(torch.min(surr1, surr2) * choice_mask).sum() / choice_mask.sum()
                        entropy = (dist.entropy() * choice_mask).sum() / choice_mask.sum()
                    else:
                        policy_loss = torch.tensor(0.0, device=device)
                        entropy = dist.entropy().mean()

                    value_loss = nn.functional.mse_loss(new_values.squeeze(-1), b_ret)
                    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    optimizer.step()

            scheduler.step()
            buffer.clear()

        # Regular Progress Log every 20 iterations
        if iteration % 20 == 0:
            elapsed = time.time() - start_time
            hours_elapsed = elapsed / 3600.0
            overall_wr = (rl_agent_wins / max(1, episodes_completed)) * 100
            recent_wr = (window_wins / max(1, window_episodes)) * 100
            sps = total_env_steps / elapsed
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"[TRAIN] Iter {iteration:5d} | Time: {hours_elapsed:4.2f}h/{target_hours:.1f}h "
                f"| Steps: {total_env_steps:8d} | Speed: {sps:5.1f} st/s "
                f"| Recent Win Rate: {recent_wr:5.1f}% ({window_wins}/{window_episodes}) "
                f"| Overall: {overall_wr:5.1f}% | LR: {current_lr:.1e}",
                flush=True,
            )
            window_wins = 0
            window_episodes = 0

        # Periodic Evaluation Tournament against Grandmaster League
        if iteration % eval_interval == 0:
            print(f"\n--- [TOURNAMENT EVALUATION @ Iteration {iteration} | {elapsed_time/3600:.2f}h] ---", flush=True)
            results = evaluate_tournament(model, device, num_games=30)
            rl_wr = results["RL-Agent (PPO)"]
            gm_wr = results["GrandmasterBot"]
            mk_wr = results["Markov-ROI"]
            ag_wr = results["Aggressive"]
            print(
                f"[EVAL RESULT] RL-Agent: {rl_wr:5.1f}% | Grandmaster: {gm_wr:5.1f}% "
                f"| Markov-ROI: {mk_wr:5.1f}% | Aggressive: {ag_wr:5.1f}%",
                flush=True,
            )

            # High-water mark checkpointing
            if rl_wr > best_eval_win_rate:
                best_eval_win_rate = rl_wr
                torch.save(model.state_dict(), "checkpoints/monopoly_ppo_best.pt")
                torch.save(model.state_dict(), "checkpoints/monopoly_ppo_final.pt")
                print(f"[RECORD] New tournament win rate record: {best_eval_win_rate:.1f}%! Saved to 'checkpoints/monopoly_ppo_best.pt' and 'monopoly_ppo_final.pt'", flush=True)
            print("-" * 65 + "\n", flush=True)

        # Regular periodic save
        if iteration % save_interval == 0:
            ckpt_path = f"checkpoints/monopoly_ppo_ep{iteration}.pt"
            torch.save(model.state_dict(), ckpt_path)
            torch.save(model.state_dict(), "checkpoints/monopoly_ppo_final.pt")

    # Final wrap-up save
    torch.save(model.state_dict(), "checkpoints/monopoly_ppo_final.pt")
    total_time = (time.time() - start_time) / 3600.0
    print(f"\n[COMPLETE] 12-Hour Overnight Training Completed Successfully!")
    print(f"Total Time: {total_time:.2f} Hours | Total Steps: {total_env_steps} | Total Games: {episodes_completed}")
    print(f"Peak Tournament Win Rate: {best_eval_win_rate:.1f}%")
    print(f"Final model weights saved to 'checkpoints/monopoly_ppo_final.pt' and 'checkpoints/monopoly_ppo_best.pt'!\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overnight Monopoly PPO League Training")
    parser.add_argument("--target-hours", "-t", type=float, default=12.5, help="Target training duration in hours (default: 12.5)")
    parser.add_argument("--episodes", "-e", type=int, default=10000, help="Max updates (default: 10000)")
    parser.add_argument("--rollout-steps", "-s", type=int, default=2048, help="Rollout steps per update (default: 2048)")
    parser.add_argument("--batch-size", "-b", type=int, default=256, help="Minibatch size for SGD (default: 256)")
    parser.add_argument("--save-interval", type=int, default=100, help="Checkpoint interval (default: 100)")
    parser.add_argument("--eval-interval", type=int, default=100, help="Evaluation tournament interval (default: 100)")
    parser.add_argument("--resume", "-r", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--no-warmup", action="store_true", help="Skip expert behavioral cloning warmup")
    args = parser.parse_args()
    sys.stdout = DualLogger("training_overnight.log")
    sys.stderr = sys.stdout

    train_ppo(
        total_episodes=args.episodes,
        target_hours=args.target_hours,
        rollout_steps=args.rollout_steps,
        batch_size=args.batch_size,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        resume_path=args.resume,
        warmup=not args.no_warmup,
    )
