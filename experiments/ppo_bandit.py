#!/usr/bin/env python3
"""
PPO on a K-armed bandit with plotting (+ sweeps).
- Trains a softmax policy (and value baseline) to maximize reward.
- Tracks & plots the probability of selecting the optimal arm over training.
- Supports multiple independent runs and aggregates mean ± std.
- NEW: Sweeps over clip epsilon and minibatch sizes to produce comparison plots.

Examples:
  # Baseline single config
  python ppo_bandit.py --arms 5 --updates 150 --batch-size 1024 --minibatch-size 256 --runs 10

  # Sweep clip eps with fixed minibatch
  python ppo_bandit.py --updates 200 --batch-size 1024 --minibatch-size 256 \
      --sweep-clip 0.1 0.2 0.3 0.4

  # Sweep minibatch sizes with fixed clip eps
  python ppo_bandit.py --updates 200 --batch-size 2048 --clip-eps 0.2 \
      --sweep-minibatch 64 128 256 512
"""

import argparse
import math
import random
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class BanditEnv:
    """K-armed bandit with Gaussian rewards N(mean[i], 1). """
    def __init__(self, means: np.ndarray):
        self.means = means.astype(np.float32)
        self.k = len(means)
        self.best_arm = int(np.argmax(means))

    def step(self, action: int) -> float:
        mean = self.means[action]
        return float(np.random.normal(loc=mean, scale=1.0))

    def sample_batch(self, policy_logits: torch.Tensor, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Vectorized batch sampling using the current policy.
        Returns:
          actions [B], rewards [B], logprobs [B]
        """
        with torch.no_grad():
            probs = torch.softmax(policy_logits, dim=-1)  # [K]
            dist = torch.distributions.Categorical(probs=probs)
            actions = dist.sample((batch_size,))  # [B]
            logprobs = dist.log_prob(actions)     # [B]
        # Rewards from env
        actions_np = actions.cpu().numpy()
        rewards = np.random.normal(loc=self.means[actions_np], scale=1.0).astype(np.float32)
        rewards = torch.from_numpy(rewards)
        return actions, rewards, logprobs


class PolicyValue(nn.Module):
    """A minimal policy+value head over a learned constant 'state' embedding."""
    def __init__(self, k: int, hidden: int = 16):
        super().__init__()
        self.emb = nn.Parameter(torch.zeros(hidden))  # learned constant context
        self.policy = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, k),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self):
        x = self.emb
        logits = self.policy(x)
        value = self.value(x).squeeze(-1)
        return logits, value


def ppo_update(
    model: PolicyValue,
    optimizer: optim.Optimizer,
    old_logprobs: torch.Tensor,
    actions: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    clip_eps: float,
    entropy_coef: float,
    value_coef: float,
    minibatch_size: int,
    epochs: int,
):
    """One PPO update over collected batch."""
    B = actions.shape[0]
    idx = torch.randperm(B)

    advantages = returns - values.detach()
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    old_logprobs = old_logprobs.detach()
    actions = actions.detach()
    returns = returns.detach()

    for _ in range(epochs):
        for start in range(0, B, minibatch_size):
            mb = idx[start:start + minibatch_size]

            logits, value_pred = model()
            dist = torch.distributions.Categorical(logits=logits)
            logprobs = dist.log_prob(actions[mb])

            ratio = torch.exp(logprobs - old_logprobs[mb])
            adv = advantages[mb]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            entropy = dist.entropy().mean()
            value_loss = (value_pred - returns[mb]).pow(2).mean()

            loss = policy_loss - entropy_coef * entropy + value_coef * value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def train_one_run(
    means: np.ndarray,
    updates: int,
    batch_size: int,
    minibatch_size: int,
    policy_lr: float,
    clip_eps: float,
    entropy_coef: float,
    value_coef: float,
    epochs: int,
    seed: int,
    hidden: int,
):
    set_seed(seed)
    env = BanditEnv(means)
    model = PolicyValue(k=len(means), hidden=hidden)
    optimizer = optim.Adam(model.parameters(), lr=policy_lr)

    best_arm = env.best_arm
    prob_correct = []

    for _ in range(updates):
        logits, values = model()
        actions, rewards, old_logprobs = env.sample_batch(logits, batch_size)
        returns = rewards  # 1-step bandit

        ppo_update(
            model=model,
            optimizer=optimizer,
            old_logprobs=old_logprobs,
            actions=actions,
            returns=returns,
            values=values.expand_as(returns),
            clip_eps=clip_eps,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
            minibatch_size=minibatch_size,
            epochs=epochs,
        )

        with torch.no_grad():
            logits, _ = model()
            probs = torch.softmax(logits, dim=-1)
            prob_correct.append(float(probs[best_arm].cpu().item()))

    return np.array(prob_correct, dtype=np.float32)


def aggregate_runs(
    means: np.ndarray,
    runs: int,
    base_seed: int,
    **kwargs,
):
    curves = []
    for r in range(runs):
        seed = base_seed + r
        curve = train_one_run(means=means, seed=seed, **kwargs)
        curves.append(curve)
    curves = np.stack(curves, axis=0)  # [runs, T]
    return curves.mean(axis=0), curves.std(axis=0)


def parse_args():
    p = argparse.ArgumentParser(description="PPO on a K-armed bandit with sweeps.")
    p.add_argument("--arms", type=int, default=5, help="Number of arms (ignored if --means provided).")
    p.add_argument("--means", type=float, nargs="*", default=None, help="Means per arm. Overrides --arms.")

    # training config
    p.add_argument("--updates", type=int, default=150, help="Number of PPO updates.")
    p.add_argument("--batch-size", type=int, default=1024, help="Samples per update.")
    p.add_argument("--minibatch-size", type=int, default=256, help="Minibatch size for PPO.")
    p.add_argument("--epochs", type=int, default=1, help="PPO epochs per update.")
    p.add_argument("--runs", type=int, default=10, help="Independent runs to average.")
    p.add_argument("--seed", type=int, default=42, help="Base RNG seed (each run adds run index).")

    # PPO hyperparams
    p.add_argument("--policy-lr", type=float, default=3e-3, help="Learning rate.")
    p.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip epsilon.")
    p.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy bonus coefficient.") #KL term here, can leave out
    p.add_argument("--value-coef", type=float, default=0.5, help="Value loss coefficient.")
    p.add_argument("--hidden", type=int, default=16, help="Hidden size for small MLP.")

    # Sweeps
    p.add_argument("--sweep-clip", type=float, nargs="*", default=None,
                   help="List of clip epsilons to compare (e.g., 0.1 0.2 0.3).")
    p.add_argument("--sweep-minibatch", type=int, nargs="*", default=None,
                   help="List of minibatch sizes to compare (e.g., 64 128 256).")

    # Outputs
    p.add_argument("--out", type=str, default="ppo_bandit_plot.png", help="Baseline plot filename.")
    p.add_argument("--out-clip", type=str, default="ppo_bandit_clip_sweep.png", help="Clip sweep plot.")
    p.add_argument("--out-minibatch", type=str, default="ppo_bandit_minibatch_sweep.png", help="Minibatch sweep plot.")
    return p.parse_args()


def maybe_make_means(args) -> np.ndarray:
    if args.means is None or len(args.means) == 0:
        means = np.linspace(-0.2, 1.0, num=args.arms, dtype=np.float32)
        np.random.shuffle(means)
    else:
        means = np.array(args.means, dtype=np.float32)
        args.arms = len(means)
    return means


def plot_with_shade(x, mean, std, label: str):
    plt.plot(x, mean, label=label)
    plt.fill_between(x, mean - std, mean + std, alpha=0.2)


def main():
    args = parse_args()
    means = maybe_make_means(args)

    # --- Baseline single config (still available) ---
    mean_prob, std_prob = aggregate_runs(
        means=means,
        runs=args.runs,
        base_seed=args.seed,
        updates=args.updates,
        batch_size=args.batch_size,
        minibatch_size=args.minibatch_size,
        policy_lr=args.policy_lr,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        epochs=args.epochs,
        hidden=args.hidden,
    )
    x = np.arange(len(mean_prob))
    plt.figure()
    plot_with_shade(x, mean_prob, std_prob, label=f"clip={args.clip_eps}, mb={args.minibatch_size}")
    plt.xlabel("PPO update")
    plt.ylabel("Probability of optimal arm")
    plt.title(f"Baseline: PPO on {args.arms}-armed bandit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=160)
    print(f"Saved baseline plot to: {args.out}")

    # --- Sweep over clip epsilons (fixed minibatch) ---
    if args.sweep_clip and len(args.sweep_clip) > 0:
        plt.figure()
        for ce in args.sweep_clip:
            mean_prob, std_prob = aggregate_runs(
                means=means,
                runs=args.runs,
                base_seed=args.seed,
                updates=args.updates,
                batch_size=args.batch_size,
                minibatch_size=args.minibatch_size,  # fixed
                policy_lr=args.policy_lr,
                clip_eps=float(ce),
                entropy_coef=args.entropy_coef,
                value_coef=args.value_coef,
                epochs=args.epochs,
                hidden=args.hidden,
            )
            x = np.arange(len(mean_prob))
            plot_with_shade(x, mean_prob, std_prob, label=f"clip={ce}")
        plt.xlabel("PPO update")
        plt.ylabel("Probability of optimal arm")
        plt.title(f"Clip sweep (minibatch={args.minibatch_size})")
        plt.legend(title="PPO clip ε")
        plt.tight_layout()
        plt.savefig(args.out_clip, dpi=160)
        print(f"Saved clip sweep plot to: {args.out_clip}")

    # --- Sweep over minibatch sizes (fixed clip epsilon) ---
    if args.sweep_minibatch and len(args.sweep_minibatch) > 0:
        plt.figure()
        for mb in args.sweep_minibatch:
            mean_prob, std_prob = aggregate_runs(
                means=means,
                runs=args.runs,
                base_seed=args.seed,
                updates=args.updates,
                batch_size=args.batch_size,
                minibatch_size=int(mb),
                policy_lr=args.policy_lr,
                clip_eps=args.clip_eps,           # fixed
                entropy_coef=args.entropy_coef,
                value_coef=args.value_coef,
                epochs=args.epochs,
                hidden=args.hidden,
            )
            x = np.arange(len(mean_prob))
            plot_with_shade(x, mean_prob, std_prob, label=f"mb={mb}")
        plt.xlabel("PPO update")
        plt.ylabel("Probability of optimal arm")
        plt.title(f"Minibatch sweep (clip={args.clip_eps})")
        plt.legend(title="Minibatch size")
        plt.tight_layout()
        plt.savefig(args.out_minibatch, dpi=160)
        print(f"Saved minibatch sweep plot to: {args.out_minibatch}")

    # Print ground-truth stats
    print(f"Best arm index (0-based): {int(np.argmax(means))}")
    print(f"Arm means: {means.tolist()}")


if __name__ == "__main__":
    main()