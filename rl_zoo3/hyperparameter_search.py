# ganz oben:
from torch import nn

# irgendwo bei den anderen Sample-Funktionen:
def sample_example_params(trial):
    # typische PPO-Suchräume; gerne anpassen
    net = trial.suggest_categorical("net_arch", ["small", "medium"])
    net_arch = dict(pi=[64, 64], vf=[64, 64]) if net == "small" else dict(pi=[256, 256], vf=[256, 256])

    return {
        "n_steps": trial.suggest_categorical("n_steps", [64, 128, 256, 512, 1024, 2048]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "gamma": trial.suggest_float("gamma", 0.95, 0.9999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 1.0),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
        "ent_coef": trial.suggest_float("ent_coef", 1e-8, 1e-2, log=True),
        "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 2.0),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True),
        "policy_kwargs": dict(activation_fn=nn.ReLU, net_arch=net_arch, ortho_init=False),
    }