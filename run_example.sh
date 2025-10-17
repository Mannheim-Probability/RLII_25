#!/bin/bash

source venv/bin/activate

python train.py --algo "ppo" --env MiniGrid-DoorKey-5x5-v0 --model DoorKey --hyperparams n_timesteps:1000 --eval-freq 100 --log-interval 100 --eval-episodes 5 --n-eval-envs 2 --seed 1 -f logs/example_project/example_1 --verbose 1

#--env AsterixNoFrameskip-v4
#python3 train.py --algo "ppo" --env MiniGrid-DoorKey-5x5-v0 --model DoorKey --save-interval 10 --frames 80000

#--env MiniGrid-DoorKey-5x5-v0 --model DoorKey