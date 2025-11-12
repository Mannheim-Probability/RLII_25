#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=1
#SBATCH -p dev_gpu_h100,dev_gpu_a100_il
#SBATCH --ntasks=1
#SBATCH --mem=700
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=1
#SBATCH --job-name=run_example
#SBATCH --output=slurms/run_example/test_setup_%j.out
#SBATCH --error=slurms/run_example/test_setup_%j.err

CUDA_VISIBLE_DEVICES=1 python train.py --algo ppo --env AsterixNoFrameskip-v4 --hyperparams n_timesteps:1000 --eval-freq 100 --log-interval 100 --eval-episodes 5 --n-eval-envs 2 --seed 1 -f logs/example_project/example_1 --verbose 1 

wait # Make sure all background processes have finished
echo "All training runs completed"