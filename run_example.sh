#!/bin/bash

source venv/bin/activate

python train.py --algo "qrdqn" --env AsterixNoFrameskip-v4 --hyperparams buffer_size:100 n_timesteps:1000 --eval-freq 100 --log-interval 100 --eval-episodes 5 --n-eval-envs 2 --seed 1 -f logs/example_project/example_1 --verbose 1