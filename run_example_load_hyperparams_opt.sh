#!/bin/bash

source .venv/bin/activate

python train.py --algo ppo  --env LunarLander-v3 -n 30000  --study-name ppo_lunarlander_nsgaii_1e6 --storage logs/ppo_lunarlander_nsgaii_1e6.log --eval-freq 1000 