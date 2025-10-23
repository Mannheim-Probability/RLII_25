#!/bin/bash

source venv/bin/activate

python train.py --algo ppo  --env LunarLander-v3 -optimize --n-trials 1000 --n-startup-trials 3 -n 30000 --n-evaluations 3  --sampler qmc --pruner median --study-name qmc_ppo_lunar_study --storage logs/test_study.log 