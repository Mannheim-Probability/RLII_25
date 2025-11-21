import warnings
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import FloatSchedule, explained_variance, obs_as_tensor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from rl_zoo3.custom_algos.ppo_corrected_4 import PPOCorrected4

from rl_zoo3.custom_buffers import TimedRolloutBuffer5

SelfPPOCorrected4_2 = TypeVar("SelfPPOCorrected4_2", bound="PPOCorrected4_2")


class PPOCorrected4_2(PPOCorrected4):
    """
    PPO_mod_advantges variant using TimedRolloutBuffer5 instead of TimedRolloutBuffer3.
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(self, *args, rollout_buffer_class=TimedRolloutBuffer5, **kwargs):
        super().__init__(*args, rollout_buffer_class=rollout_buffer_class, **kwargs)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: TimedRolloutBuffer5,
        n_rollout_steps: int,
    ) -> bool:
        return super().collect_rollouts(
            env, 
            callback, 
            rollout_buffer, 
            n_rollout_steps
            )
