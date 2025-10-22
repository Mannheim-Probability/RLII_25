#!/bin/bash

source .venv/bin/activate

# python train.py --algo ppo  --env CartPole-v1 -optimize --n-trials 10 --n-startup-trials 3 -n 30000 --n-evaluations 3  --sampler random --pruner median --study-name test_study --storage logs/test_study.log 

# python train.py --algo ppo  --env CartPole-v1 -optimize --n-trials 10 --n-startup-trials 3 -n 30000 --n-evaluations 3  --sampler nsgaii --pruner auto --study-name test_study --storage logs/test_study.log 

# === NSGA-II HPO for PPO on LunarLander (discrete) ===
# Adjust TRIALS/EVALS if you need a faster or more thorough run.
ENV_NAME="LunarLander-v3"
ALGO="ppo_changed_before_normalization"
TRIALS=30            # reasonable budget; try 10 for a quick dry-run
STARTUP_TRIALS=10    # random init for NSGA-II
TIMESTEPS=1000000    # 1e6 total env steps per trial
EVALS=8              # evaluate progress a few times during training
SAMPLER="nsgaii"
PRUNER="median"      # robust early-stopping; use "auto" if your train.py supports it well
STUDY="ppo_changed_before_normalization_lunarlander_nsgaii_1e6"
STORAGE="logs/${STUDY}.log"

python train.py \
  --algo "$ALGO" \
  --env "$ENV_NAME" \
  -optimize \
  --n-trials $TRIALS \
  --n-startup-trials $STARTUP_TRIALS \
  -n $TIMESTEPS \
  --n-evaluations $EVALS \
  --sampler $SAMPLER \
  --pruner $PRUNER \
  --study-name "$STUDY" \
  --storage "$STORAGE"

# --- Optional: tiny smoke test (uncomment to use) ---
# TRIALS=6; STARTUP_TRIALS=3; TIMESTEPS=150000; EVALS=3; STUDY="ppo_lunarlander_nsgaii_smoke"; STORAGE="logs/${STUDY}.log"
# python train.py --algo "$ALGO" --env "$ENV_NAME" -optimize --n-trials $TRIALS --n-startup-trials $STARTUP_TRIALS -n $TIMESTEPS --n-evaluations $EVALS --sampler $SAMPLER --pruner $PRUNER --study-name "$STUDY" --storage "$STORAGE"