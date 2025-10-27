# RL II Lecture HWS25 University of Mannheim

This is the GitHub Repository for the lecture Reinforcement Learning II in the fall semester of 2025 at the University of Mannheim. It is forked and modified from the [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) Repository. Only the core functionalities of the repository are contained, such that it will be easy to get accustomed to the workings. The interested student is encouraged to explore the original repo and use any further tools they might find useful in their branch.

RL Baselines3 Zoo is a training framework for Reinforcement Learning (RL), using [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3). It provides scripts for training, evaluating agents, tuning hyperparameters, plotting results, and recording videos.

In addition, it includes a collection of tuned hyperparameters for common environments and RL algorithms, and agents trained with those settings (removed).

## Documentation

Documentation is available online: [https://rl-baselines3-zoo.readthedocs.io/](https://rl-baselines3-zoo.readthedocs.io) (this includes removed features found in the original repo as well).

## Videos of the Lectures

We will be recording some of the lectures due to conflicting schedules of some of the students.

[Lecture 2 (part of it)](https://unimannheimde.sharepoint.com/:v:/s/TeamStochastik/EV8z09NXh3BImPhA04YHYTABSvvwrErPzpOvO5xNRrMtRA?e=mZpRLr)

[Lecture 3](https://unimannheimde.sharepoint.com/:v:/s/TeamStochastik/EVV-H8FOI3lImgl_YrISeC4BUpc05JXElaJq5PRlraorig?e=q0QEg9)

## Installation

This installation is written and tested for MacOS. For Windows, the steps are similar but might involve slightly different programs and commands.

Install [pyenv](https://github.com/pyenv/pyenv) and set the python version in the directory to 3.9.19. Then clone the github project:

```
git clone https://github.com/Mannheim-Probability/RL-Mannheim
``` 

Now create a branch with your name and check it out. Then create a virtual environment:

```
python -m venv venv
```

Select it for your workspace, and activate it via:

```
source venv/bin/activate
```

Install the necessary requirements:

```
pip install swig
pip install -r requirements.txt
```

or use poetry install after installing swig via pip (already configured).

## Train an Agent

The hyperparameters for each environment are defined in `hyperparameters/algo_name.yml`.

If the environment exists in this file, then you can train an agent using:
```
python train.py --algo algo_name --env env_id
```

Evaluate the agent every 10000 steps using 10 episodes for evaluation (using only one evaluation env):
```
python train.py --algo sac --env HalfCheetahBulletEnv-v0 --eval-freq 10000 --eval-episodes 10 --n-eval-envs 1
```

More examples are available in the [documentation](https://rl-baselines3-zoo.readthedocs.io).

Be careful: You will not be able to train any pybullet or panda robot environments since there is an issue for installing pybullet (which both types of environments use) on MacOS!

## Integrations

The RL Zoo has some integration with other libraries/services like Weights & Biases for experiment tracking or Hugging Face for storing/sharing trained models. You can find out more in the [dedicated section](https://rl-baselines3-zoo.readthedocs.io/en/master/guide/integrations.html) of the documentation (largely removed).

## Plot Scripts

Please see the [dedicated section](https://rl-baselines3-zoo.readthedocs.io/en/master/guide/plot.html) of the documentation (removed).

## Enjoy a Trained Agent

The original repository comes with some pretrained agents (removed). You can in principle use any agent you trained yourself with the following commands or add one of the trained agents from Stable Baselines.

If the trained agent exists, then you can see it in action using:
```
python enjoy.py --algo algo_name --env env_id
```

For example, enjoy A2C on Breakout during 5000 timesteps:
```
python enjoy.py --algo a2c --env BreakoutNoFrameskip-v4 --folder rl-trained-agents/ -n 5000
```

## Hyperparameters Tuning

Please see the [dedicated section](https://rl-baselines3-zoo.readthedocs.io/en/master/guide/tuning.html) of the documentation.

## Custom Configuration

Please see the [dedicated section](https://rl-baselines3-zoo.readthedocs.io/en/master/guide/config.html) of the documentation (removed).

## Current Collection: 200+ Trained Agents!

Everything that has to do with the "benchmarks" from Stable Baselines 3 Zoo has been removed (trained agents, scripts to replicate the training, tuned hyperparameter settings, data, videos of trained agents). Since it was only trained on one seed each the data is not very expressive anyways.

## Colab Notebook: Try it Online!

You can train agents online using [Colab notebook](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/rl-baselines-zoo.ipynb).

### Passing arguments in an interactive session

The zoo is not meant to be executed from an interactive session (e.g: Jupyter Notebooks, IPython), however, it can be done by modifying `sys.argv` and adding the desired arguments.

*Example*
```python
import sys
from rl_zoo3.train import train

sys.argv = ["python", "--algo", "ppo", "--env", "MountainCar-v0"]

train()
```

## Tests

To run tests (removed), first install pytest, then:
```
make pytest
```

Same for type checking with pytype:
```
make type
```

## Citing the Project

To cite this repository in publications (might be relevant if we come up with a paper-worthy idea):

```bibtex
@misc{rl-zoo3,
  author = {Raffin, Antonin},
  title = {RL Baselines3 Zoo},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/DLR-RM/rl-baselines3-zoo}},
}
```
