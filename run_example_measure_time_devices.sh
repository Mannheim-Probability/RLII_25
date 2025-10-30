#!/bin/bash

source venv/bin/activate

run_and_time() {
    local name="$1"
    shift
    local start=$(date +%s)
    "$@"  # run the actual command
    local end=$(date +%s)
    local elapsed=$((end - start))
    echo "$name took: ${elapsed}s"
}

run_and_time "Execution on CPU" python train.py --algo "ppo" --env AsterixNoFrameskip-v4 --hyperparams n_timesteps:1000000 --eval-freq 50000 --log-interval 5000 --eval-episodes 5 --n-eval-envs 2 --seed 1 -f logs/example_project/example_1 --verbose 1 --device cpu
run_and_time "Execution on GPU" python train.py --algo "ppo" --env AsterixNoFrameskip-v4 --hyperparams n_timesteps:1000000 --eval-freq 50000 --log-interval 5000 --eval-episodes 5 --n-eval-envs 2 --seed 1 -f logs/example_project/example_1 --verbose 1 --device mps
run_and_time "Execution on automatic device" python train.py --algo "ppo" --env AsterixNoFrameskip-v4 --hyperparams n_timesteps:10000 --eval-freq 5000 --log-interval 5000 --eval-episodes 5 --n-eval-envs 2 --seed 1 -f logs/example_project/example_1 --verbose 1