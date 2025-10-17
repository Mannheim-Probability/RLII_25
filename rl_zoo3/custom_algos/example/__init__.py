# rl_zoo3/custom_algos/example/__init__.py
from .example import EXAMPLE
from rl_zoo3.custom_algos.example.policies import CnnPolicy, MlpPolicy, MultiInputPolicy

__all__ = ["EXAMPLE", "CnnPolicy", "MlpPolicy", "MultiInputPolicy",]