#!/bin/bash

source .venv/bin/activate

python train.py --algo "ppo" --env AsterixNoFrameskip-v4 --hyperparams n_timesteps:1000 --eval-freq 100 --log-interval 100 --eval-episodes 5 --n-eval-envs 2 --seed 1 -f logs/homework_1/example_baseline --verbose 1

# python train.py --algo ppo --env CartPole-v1 -n 5e4 \
#   --eval-freq 5000 --eval-episodes 5 --n-eval-envs 2 \
#   -f logs/smoke_test --seed 1 --verbose 1

# python enjoy.py --algo "ppo" --env AsterixNoFrameskip-v4 -f logs/example_project/example_1