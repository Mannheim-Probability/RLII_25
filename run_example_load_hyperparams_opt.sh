#!/bin/bash

source .venv/bin/activate

python train.py --algo ppo_mod_advantages  --env LunarLander-v3 -n 1000000  --study-name ppo_mod_ad_lunarlander_nsgaii_1e6 --storage logs/ppo_mod_ad_lunarlander_nsgaii_1e6.log --eval-freq 1000 