#!/bin/bash

source venv/bin/activate

python train.py --algo ppo_changed_before_normalization  --env LunarLander-v3 -optimize --n-trials 1000 --n-startup-trials 3 -n 1000000 --n-evaluations 3  --sampler qmc --pruner median --study-name qmc_before_nromalization_ppo_lunar_study --storage logs/test_study.log 