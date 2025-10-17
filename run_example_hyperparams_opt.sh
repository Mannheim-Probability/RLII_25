#!/bin/bash

source venv/bin/activate

python train.py --algo ppo  --env CartPole-v1 -optimize --n-trials 10 --n-startup-trials 3 -n 30000 --n-evaluations 3  --sampler random --pruner median --study-name test_study --storage logs/test_study.log 