#!/bin/bash

source venv/bin/activate

python train.py --algo ppo  --env LunarLander-v3 -n 30000  --study-name qmc_ppo_lunar_study --storage logs/test_study.log  --eval-freq 1000 --seed 123
python train.py --algo ppo_changed_before_normalization  --env LunarLander-v3 -n 30000  --study-name qmc_before_nromalization_ppo_lunar_study --storage logs/test_study.log  --eval-freq 1000 --seed 123
#python train.py --algo ppo_mod_advantages  --env LunarLander-v3 -n 30000  --study-name qmc_before_nromalization_ppo_lunar_study --storage logs/test_study.log  --eval-freq 1000 --seed 123
#python train.py --algo ppo_mod_sampling  --env LunarLander-v3 -n 30000  --study-name qmc_before_nromalization_ppo_lunar_study --storage logs/test_study.log  --eval-freq 1000 --seed 123