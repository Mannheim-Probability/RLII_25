source .venv/bin/activate 

python rl_zoo3/plots/plot_train.py \
    -a ppo \
    -e LunarLander-v3 \
    -f logs \
    -x steps -w 100 \