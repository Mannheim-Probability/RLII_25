#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=2
#SBATCH -p dev_gpu_h100,dev_gpu_a100_il
#SBATCH --ntasks=4
#SBATCH --mem=4000
#SBATCH --gres=gpu:4
#SBATCH --job-name=run_example
#SBATCH --output="(/path/to/workspace/)slurms/run_example/example_3_%j.out"
#SBATCH --error="(/path/to/workspace/)slurms/run_example/example_3_%j.err"
# output and error files may be directed to WORKSPACE or $HOME as needed

# Does not currently work with hyperparameter optimization!

source venv/bin/activate

WORKSPACE="/path/to/workspace"

# If using different -f or -tb paths, make sure to create the corresponding directories
mkdir -p "$WORKSPACE/logs"
mkdir -p "$WORKSPACE/wandb"
mkdir -p "$TMPDIR/logs"
mkdir -p "$TMPDIR/wandb"

export WANDB_DIR="$TMPDIR/wandb"
export WANDB_DATA_DIR="$TMPDIR/wandb"
export WANDB_CACHE_DIR="$TMPDIR/wandb/.cache"
export WAND_RUN_DIR="$TMPDIR/wandb"

cleanup() {
    echo "===================="
    echo "Cleanup initiated..."
    echo "===================="
    
    echo "Copying results from $TMPDIR to $WORKSPACE..."

    # If using different -f or -tb paths, make sure to include them
    rsync -av "$TMPDIR/logs/" "$WORKSPACE/logs/"
    rsync -av "$TMPDIR/wandb/" "$WORKSPACE/wandb/"

    echo "Results saved to $WORKSPACE"
}

trap cleanup SIGTERM EXIT SIGINT 

algo="dqn"
envs=("NameThisGameNoFrameskip-v4")
seeds=("1" "2" "3" "4")

# Assuming you have 4 GPUs available
for i in "${!seeds[@]}"; do

    seed=${seeds[$i]}

    for j in "${!envs[@]}"; do

        env=${envs[$j]}
        gpu_id=$((i+j))

        # Use CUDA_VISIBLE_DEVICES to specify which GPU to use for each task
        CUDA_VISIBLE_DEVICES=$gpu_id python train.py --algo $algo --env $env --hyperparams n_timesteps:1000 --eval-freq 25 --log-interval 25 --eval-episodes 2 --n-eval-envs 1 --seed $seed -f "$TMPDIR/logs/example_project/example_3" --verbose 1 --uuid --track --wandb-project-name "Cluster_testing" --wandb-entity "RL2_2025" -tags test_cluster_workspaces  &

    done
done

wait # Make sure all background processes have finished
echo "All training runs completed"