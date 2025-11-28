#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH -p gpu_h100,gpu_a100_il,gpu_h100_il
#SBATCH --ntasks=4
#SBATCH --mem=235000
#SBATCH --gres=gpu:4
#SBATCH --job-name=ppo_mod_atari
#SBATCH --output=slurms/ppo_mod_atari/ppo_%j.out
#SBATCH --error=slurms/ppo_mod_atari/ppo_%j.err

algo="ppo"
envs=("NameThisGameNoFrameskip-v4")
seeds=("565138552" "874004163" "1866447363" "2514086935")

# Assuming you have 4 GPUs available
for i in "${!seeds[@]}"; do

    seed=${seeds[$i]}

    for j in "${!envs[@]}"; do

        env=${envs[$j]}
        gpu_id=$((i+j))

        # Use CUDA_VISIBLE_DEVICES to specify which GPU to use for each task
        CUDA_VISIBLE_DEVICES=$gpu_id python train.py --algo $algo --env $env --hyperparams n_timesteps:20000000 --eval-freq 50000 --log-interval 10000 --eval-episodes 10 --n-eval-envs 10 --seed $seed -f "logs/ppo_mod_atari/ppo" --verbose 1 --uuid --track --wandb-project-name "NameThisGame" --wandb-entity "RL2_2025" -tags run_ppo_configs_atari &

    done
done

wait # Make sure all background processes have finished
echo "All training runs completed"