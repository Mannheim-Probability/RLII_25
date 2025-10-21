#!/usr/bin/env bash
set -euo pipefail

# --- Activate your virtualenv (tries .venv first, then venv) ---
if [ -d ".venv" ]; then
  source .venv/bin/activate
elif [ -d "venv" ]; then
  source venv/bin/activate
else
  echo "No .venv/ or venv/ found. Create one first (python -m venv .venv)."
  exit 1
fi

# --- HPO run config ---
ENV_NAME="LunarLander-v3"
ALGO="ppo_mod_advantages"

TRIALS=30            # total HPO trials (the enqueued seed counts toward this)
STARTUP_TRIALS=10    # initial random trials to seed the population
TIMESTEPS=1000000    # 1e6 env steps per trial
EVALS=8              # intermediate evaluations per trial

SAMPLER="nsgaii"
PRUNER="median"      # robust default; "auto" works too in many setups

STUDY="ppo_modad_lunarlander_nsgaii_1e6_withinit"
STORAGE="logs/${STUDY}.log"   # Optuna Journal storage file
mkdir -p "$(dirname "$STORAGE")"

echo "==> Enqueuing a PPO seed (matching sample_ppo_params names) into study '${STUDY}'"

# Notes on the seed below:
# - n_steps_pow=11  -> 2**11 = 2048 rollout steps
# - batch_size_pow=8 -> 2**8 = 256 batch size
# - one_minus_gamma=0.01 -> gamma = 0.99
# - one_minus_gae_lambda=0.05 -> gae_lambda = 0.95
python - <<'PY'
import os
import optuna
from optuna.storages import JournalStorage, JournalFileStorage

study_name = os.environ.get("STUDY")
storage_path = os.environ.get("STORAGE")

storage = JournalStorage(JournalFileStorage(storage_path))
# Single-objective setup (maximize reward). If you later add a 2nd objective, change directions accordingly.
study = optuna.create_study(
    study_name=study_name,
    directions=["maximize"],
    storage=storage,
    load_if_exists=True,
)

seed_params = {
    "n_steps_pow": 10,             # 2**10 = 1024
    "batch_size_pow": 6,           # 2**6  = 64
    "one_minus_gamma": 0.001,      # gamma = 0.999
    "one_minus_gae_lambda": 0.02,  # gae_lambda = 0.98
    "learning_rate": 3e-4,         # YAML didn’t specify; 3e-4 is the usual PPO default
    "ent_coef": 0.01,
    "clip_range": 0.2,             # typical default unless your repo sets another
    "n_epochs": 4,
    "max_grad_norm": 0.5,          # standard default
    "net_arch": "small",           # assuming "small" ≈ (64,64) in your converter
    "activation_fn": "tanh",
}
study.enqueue_trial(seed_params)
print("Enqueued seed trial:", seed_params)
PY

echo "==> Starting NSGA-II HPO for ${ALGO} on ${ENV_NAME}"
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

echo "==> Done. Study stored at: $STORAGE"