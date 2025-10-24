#!/bin/bash

source venv/bin/activate

python train.py --algo ppo  --env LunarLander-v3 -n 1000000  --study-name Lunar_test_study --storage logs/Lunar_test_study.log --eval-freq 5000 --track --wandb-project-name "LunarLanderVergleich" --wandb-entity "RL2_2025" -tags "optimized"
