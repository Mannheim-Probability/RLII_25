#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=1
#SBATCH -p dev_gpu_h100,dev_gpu_a100_il
#SBATCH --ntasks=4
#SBATCH --mem=4000
#SBATCH --gres=gpu:4
#SBATCH --job-name=run_example
#SBATCH --output=slurms/run_example/example_3_%j.out
#SBATCH --error=slurms/run_example/example_3_%j.err

algo="dqn"
envs=("NameThisGameNoFrameskip-v4")
seeds=("1" "2" "3" "4")

# Assuming you have 2 GPUs available
for i in "${!seeds[@]}"; do

    seed=${seeds[$i]}

    for j in "${!envs[@]}"; do

        env=${envs[$j]}
        gpu_id=$((i+j))

        # Use CUDA_VISIBLE_DEVICES to specify which GPU to use for each task
        CUDA_VISIBLE_DEVICES=$gpu_id python train.py --algo $algo --env $env --hyperparams n_timesteps:100 --eval-freq 25 --log-interval 25 --eval-episodes 2 --n-eval-envs 1 --seed $seed -f "logs/example_project/example_3" --verbose 1 --uuid --track --wandb-project-name "Cluster_testing" --wandb-entity "RL2_2025" -tags test_cluster_wandb  &

    done
done

wait # Make sure all background processes have finished
echo "All training runs completed"