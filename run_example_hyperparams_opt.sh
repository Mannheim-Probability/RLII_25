#!/bin/bash

source venv/bin/activate

python train.py --algo ppo_mod_gae --env LunarLander-v3 -optimize --n-trials 100 --hyperparams gamma:0.998 --n-startup-trials 15 -n 1000000 --n-evaluations 10 --sampler auto --pruner median --study-name lunarlander_0998_ppo_mod_gae --storage logs/lunarlander_0998_ppo_mod_gae.log --device mps
