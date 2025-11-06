#!/bin/bash

source venv/bin/activate

python train.py --algo ppo --env LunarLander-v3 -optimize --n-trials 100 --hyperparams gamma:0.998 --n-startup-trials 15 -n 1000000 --n-evaluations 10  --sampler tpe --pruner median --study-name lunarlander_ppo --storage logs/lunarlander_ppo.log 