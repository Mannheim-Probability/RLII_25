from collections.abc import Generator
from typing import Optional, Union, NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces

from rl_zoo3.custom_buffers.timed_rollout_buffer import TimedRolloutBuffer, TimedRolloutBufferSampling





class TimedRolloutBufferGaeTau(TimedRolloutBuffer):
    """
    Rollout buffer that also saves time step of transitions during the rollout and uses a modified GAE calculation with stopping time estimator.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """
    T: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 0.95,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        if gae_lambda >= 1.0:
            raise ValueError(
            "Invalid configuration: gae_lambda = 1.0 leads to division by zero in the modified GAE weighting formula."
            )   
        
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs)
        
        

    def reset(self) -> None:
        self.T = np.zeros(self.n_envs, dtype=np.float32)
        super().reset()

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """
        Uses stopping time Generalized Advantage Estimation to compute the advantage. 
        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()  # type: ignore[assignment]

        last_gae_lam = 0


        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                self.T = self.times[step] + 1
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = (
                delta
                + (1 - self.gae_lambda ** (self.T - self.times[step] - 1))
                / (1 - self.gae_lambda ** (self.T - self.times[step]))
                * self.gamma
                * self.gae_lambda
                * next_non_terminal
                * last_gae_lam
            )
            self.advantages[step] = last_gae_lam


            if step > 0:
                self.T = np.where(self.episode_starts[step], self.times[step - 1] + 1, self.T)

        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values




class TimedRolloutBufferSamplingGaeTau(TimedRolloutBufferSampling):
    """
    Rollout buffer that also saves time step of transitions during the rollout and uses a sampling procedure that samples transitions based on their time step.
    Furthermore, it uses a modified GAE calculation with stopping time estimator.
    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """
    T: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 0.95,
        gamma: float = 0.99,
        n_envs: int = 1,
        weighted_sampling: bool = True,
    ):
        if gae_lambda >= 1.0:
            raise ValueError(
            "Invalid configuration: gae_lambda = 1.0 leads to division by zero in the modified GAE weighting formula."
            )   
        
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs, weighted_sampling)
        
        

    def reset(self) -> None:
        self.T = np.zeros(self.n_envs, dtype=np.float32)
        super().reset()

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """
        Uses stopping time Generalized Advantage Estimation to compute the advantage. 
        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()  # type: ignore[assignment]

        last_gae_lam = 0


        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                self.T = self.times[step] + 1
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = (
                delta
                + (1 - self.gae_lambda ** (self.T - self.times[step] - 1))
                / (1 - self.gae_lambda ** (self.T - self.times[step]))
                * self.gamma
                * self.gae_lambda
                * next_non_terminal
                * last_gae_lam
            )
            self.advantages[step] = last_gae_lam


            if step > 0:
                self.T = np.where(self.episode_starts[step], self.times[step - 1] + 1, self.T)

        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values
