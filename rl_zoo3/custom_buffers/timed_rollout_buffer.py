from collections.abc import Generator
from typing import Optional, Union, NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecNormalize

from stable_baselines3.common.buffers import RolloutBuffer


class TimedRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    times: th.Tensor


class TimedRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer that also saves time step of transitions during the rollout. 

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    times: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs)
        
        self.reset()

    def reset(self) -> None:
        self.times = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
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
        self.times[self.pos] = np.array(time)
        super().add(obs, action, reward, episode_start, value, log_prob)

    def get(self, batch_size: Optional[int] = None) -> Generator[TimedRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "times",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> TimedRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.times[batch_inds].flatten(),
        )
        return TimedRolloutBufferSamples(*tuple(map(self.to_torch, data)))


class TimedRolloutBufferSampling(TimedRolloutBuffer):
    """
    Rollout buffer that also saves time step of transitions during the rollout and uses a sampling procedure that samples transitions based on their time step.

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
        super().add(obs, action, reward, episode_start, time, value, log_prob)

    def get(self, batch_size=None):
        if not self.weighted_sampling:
            yield from super().get(batch_size)
            return

        assert self.full

        if not self.generator_ready:
            for tensor in [
                "observations", "actions", "values",
                "log_probs", "advantages", "returns", "times", "prob_weights"
            ]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        N = self.buffer_size * self.n_envs

        weights = self.prob_weights
        weights = weights if weights.ndim == 1 else weights.flatten()

        if weights.sum() == 0:
            weights = np.ones_like(weights)

        p = weights / weights.sum()

        start_idx = 0
        while start_idx < N:
            inds = np.random.choice(N, batch_size, replace=True, p=p)
            yield self._get_samples(inds)
            start_idx += batch_size




class TimedRolloutBufferEpisodic(TimedRolloutBuffer):
    """
    Episodic Rollout buffer:
    - speichert (pos)-Indices gruppiert nach Episodes
    - key = (env_idx, episode_id)
    - value = [pos_0, pos_1, ..., pos_T]
    - get() sampled ganze Episoden, mapped sie über swap_and_flatten
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1.0,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space,
                         device, gae_lambda, gamma, n_envs)
        self.episodes_per_env = None
        self.episode_to_indices = {}
        self.reset()

    def reset(self, episodes_per_env: Optional[np.ndarray] = None) -> None:
        self.episode_to_indices = {}
        if episodes_per_env is not None:
            self.episodes_per_env = np.asarray(episodes_per_env, dtype=int)
        super().reset()

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        time: np.ndarray,
        episode_id: np.ndarray,  # shape (n_envs,)
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:

        current_pos = self.pos

        for env_idx, ep_id in enumerate(episode_id):

            if self.episodes_per_env is not None:
                # gültig: 0 ... K-1
                if ep_id >= self.episodes_per_env[env_idx]:
                    continue

            key = (env_idx, int(ep_id))

            if key not in self.episode_to_indices:
                self.episode_to_indices[key] = []

            self.episode_to_indices[key].append(current_pos)

        super().add(obs, action, reward, episode_start, time, value, log_prob)

    def get(self, n_episodes_to_batch: Optional[int] = None):

        assert self.full, "Rollout buffer not full"

        # SB3 flatten:
        if not self.generator_ready:
            tensor_fields = [
                "observations", "actions", "values",
                "log_probs", "advantages",
                "returns", "times",
            ]
            for name in tensor_fields:
                self.__dict__[name] = self.swap_and_flatten(self.__dict__[name])
            self.generator_ready = True

        all_keys = list(self.episode_to_indices.keys())
        n_total = len(all_keys)
        assert n_total > 0

        if n_episodes_to_batch is None:
            n_episodes_to_batch = n_total

        perm = np.random.permutation(n_total)
        start = 0

        while start < n_total:
            batch_keys = [all_keys[i] for i in perm[start:start+n_episodes_to_batch]]

            flat_indices = []
            for (env_idx, ep_id) in batch_keys:
                for pos in self.episode_to_indices[(env_idx, ep_id)]:
                    flat_idx = env_idx * self.buffer_size + pos
                    flat_indices.append(flat_idx)

            batch_inds = np.array(flat_indices, dtype=np.int64)

            yield self._get_samples(batch_inds)

            start += n_episodes_to_batch

    def _get_samples(self, batch_inds: np.ndarray, env=None):
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds],
            self.log_probs[batch_inds],
            self.advantages[batch_inds],
            self.returns[batch_inds],
            self.times[batch_inds],
        )
        return TimedRolloutBufferSamples(*tuple(map(self.to_torch, data)))
