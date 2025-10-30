#!/bin/bash

source venv/bin/activate

python train.py --algo ppo  --env LunarLander-v3 -optimize --n-trials 5 --n-startup-trials 3 -n 100000 --n-evaluations 5 --sampler random --pruner median --study-name Lunar_test_study --storage logs/Lunar_test_study.log 