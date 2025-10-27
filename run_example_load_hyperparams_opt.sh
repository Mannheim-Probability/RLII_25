#!/bin/bash

source venv/bin/activate

python train.py --algo ppo  --env LunarLander-v3 -n 1000000 --hyperparams gamma:0.998 --study-name lunarlander_ppo --storage logs/lunarlander_ppo.log --eval-freq 5000 --eval-episodes 5 --track --wandb-project-name "LunarLanderVergleich" --wandb-entity "RL2_2025" -tags "optimized_params"
