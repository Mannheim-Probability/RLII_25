#!/bin/bash

source venv/bin/activate

python train.py --algo ppo_no_clipping --env LunarLander-v3 -optimize --n-trials 100 --hyperparams gamma:0.9995 --n-startup-trials 15 -n 1000000 --n-evaluations 10  --sampler gp --pruner median --study-name lunarlander_ppo_no_clip --storage logs/lunarlander_ppo_no_clip.log 