#!/bin/bash

source venv/bin/activate

python train.py --algo ppo --env LunarLander-v3 -n 1000000 --track --wandb-project-name "LunarLanderVergleich" --wandb-entity "RL2_2025" -tags test_run_Arne_PPOCorrected  --eval-freq 1000 
