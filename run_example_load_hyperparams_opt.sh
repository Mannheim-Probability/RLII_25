#!/bin/bash

source venv/bin/activate

python train.py --algo ppo  --env CartPole-v1 -n 30000  --study-name test_study --storage logs/test_study.log --wandb-project-name RL_2025 -tags test --track --eval-freq 1000 