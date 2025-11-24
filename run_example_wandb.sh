#!/bin/bash

source venv/bin/activate

python train.py --algo ppo --env CartPole-v1 -n 100000 --track --wandb-project-name "test" --wandb-entity "RL2_2025" -tags test_run_daniel  --eval-freq 1000 


python train.py --algo ppo --env MountainCarContinuous-v0 -n 1000000 --track --wandb-project-name "RL_Vergleich" --wandb-entity "simeon-buettner-university-of-mannheim" -tags test_run_1  --eval-freq 500

python train.py --algo ppo_mod_gae_2 --env MountainCarContinuous-v0 -n 1000000 --track --wandb-project-name "RL_Vergleich" --wandb-entity "simeon-buettner-university-of-mannheim" -tags test_run_1  --eval-freq 500 --verbose 0 --seed 0
python train.py --algo ppo_mod_gae_3 --env MountainCarContinuous-v0 -n 1000000 --track --wandb-project-name "RL_Vergleich" --wandb-entity "simeon-buettner-university-of-mannheim" -tags test_run_1  --eval-freq 500
python train.py --algo ppo_mod_gae --env MountainCarContinuous-v0 -n 1000000 --track --wandb-project-name "RL_Vergleich" --wandb-entity "simeon-buettner-university-of-mannheim" -tags test_run_1  --eval-freq 500

for algo in ppo ppo_mod_gae_2 ppo_mod_gae_3 ppo_mod_gae
do
  for seed in 0 1 2 3 4
  do
    echo "Running algo=$algo seed=$seed"

    python train.py \
      --algo $algo \
      --env MountainCarContinuous-v0 \
      -n 1000000 \
      --track \
      --wandb-project-name "RL_Vergleich" \
      --wandb-entity "simeon-buettner-university-of-mannheim" \
      -tags test_run_vgl \
      --eval-freq 250 \
      --verbose 0 \
      --seed $seed
  done
done