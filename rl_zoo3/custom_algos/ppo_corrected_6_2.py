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
from rl_zoo3.custom_algos.ppo_corrected_6 import PPOCorrected6

from rl_zoo3.custom_buffers import TimedRolloutBuffer6

SelfPPOCorrected4_2 = TypeVar("SelfPPOCorrected6_2", bound="PPOCorrected6_2")


class PPOCorrected6_2(PPOCorrected6):
    """
    PPO_mod_sampling variant using TimedRolloutBuffer6 instead of TimedRolloutBuffer4.
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(self, *args, rollout_buffer_class=TimedRolloutBuffer6, **kwargs):
        super().__init__(*args, rollout_buffer_class=rollout_buffer_class, **kwargs)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: TimedRolloutBuffer6,
        n_rollout_steps: int,
    ) -> bool:
        return super().collect_rollouts(
            env, 
            callback, 
            rollout_buffer, 
            n_rollout_steps
            )
