from collections.abc import Generator
from typing import Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples


class TimedRolloutBuffer2(RolloutBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    prob_weights: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        weighted_sampling: bool = True,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs)
        self.weighted_sampling = weighted_sampling
        self.reset()

    def reset(self) -> None:
        self.prob_weights = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        super().reset()

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        time: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param time: Time step
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        self.prob_weights[self.pos] = np.pow(self.gamma, time)
        super().add(obs, action, reward, episode_start, value, log_prob)

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        if not self.weighted_sampling:
            yield from super().get(batch_size)
            return
        assert self.full, ""
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            indices = np.random.choice(
                self.buffer_size * self.n_envs,
                batch_size,
                # replace=False,
                p=self.prob_weights.flatten() / np.sum(self.prob_weights),
            )
            yield self._get_samples(indices)
            start_idx += batch_size
