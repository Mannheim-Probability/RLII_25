import os
from copy import deepcopy
from functools import wraps
from threading import Thread
from typing import  Any, Optional, Union

import optuna
from sb3_contrib import TQC
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, EventCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization

import gymnasium as gym
import numpy as np

from stable_baselines3.common.evaluation import evaluate_policy
from rl_zoo3.custom_evaluation.evaluation_custom import evaluate_policy_custom


class EvalCallbackCustom(EvalCallback):
    """
    Custom version of SB3's EvalCallback that uses
    the discounted return and instead of the standard (undiscounted) return.

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
        use_discounted_return: bool = True,
        gamma: float = 0.999,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
            callback_on_new_best=callback_on_new_best,
            callback_after_eval=callback_after_eval,
            warn=warn, 
        )

        # Custom attributes
        self.use_discounted_return = use_discounted_return
        self.gamma = gamma
        self.evaluations_discounted_results: list[list[float]] = []

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.verbose >= 1:
                metric_type = "discounted" if self.use_discounted_return else "undiscounted"
                print(f"Evaluating using {metric_type} returns (gamma={self.gamma})")

            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_discounted_rewards, episode_lengths = evaluate_policy_custom(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
                gamma = self.gamma,
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_discounted_rewards, list)
                assert isinstance(episode_lengths, list)

                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_discounted_results.append(episode_discounted_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    discounted_results=self.evaluations_discounted_results,
                    ep_lengths=self.evaluations_length,
                    meta_use_discounted=int(self.use_discounted_return),
                    **kwargs,  # type: ignore[arg-type]
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_discounted_reward, std_discounted_reward = np.mean(episode_discounted_rewards), np.std(episode_discounted_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)

            if self.use_discounted_return:
                self.last_mean_reward = float(mean_discounted_reward)
            else:
                self.last_mean_reward = float(mean_reward)
            

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_discounted_reward={mean_discounted_reward:.2f} +/- {std_discounted_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_discounted_reward", float(mean_discounted_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)
            self.logger.record("eval/std_reward", std_reward)
            self.logger.record("eval/std_discounted_reward", std_discounted_reward)
            
            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if self.use_discounted_return:
                if mean_discounted_reward > self.best_mean_reward:
                    if self.verbose >= 1:
                        print("New best mean discounted return!")
                    if self.best_model_save_path is not None:
                        self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                    self.best_mean_reward = float(mean_discounted_reward)
                    # Trigger callback on new best model, if needed
                    if self.callback_on_new_best is not None:
                        continue_training = self.callback_on_new_best.on_step()
            else:
                if mean_reward > self.best_mean_reward:
                    if self.verbose >= 1:
                        print("New best mean return!")
                    if self.best_model_save_path is not None:
                        self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                    self.best_mean_reward = float(mean_reward)
                    # Trigger callback on new best model, if needed
                    if self.callback_on_new_best is not None:
                        continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training


class TrialEvalCallbackCustom(EvalCallbackCustom):
    """
    Callback used for evaluating and reporting a trial based on discounted return.
    """
    def __init__(
        self,
        eval_env: VecEnv,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
        best_model_save_path: Optional[str] = None,
        log_path: Optional[str] = None,
        use_discounted_return: bool = True,
        gamma: float = 0.999,
    ) -> None:
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
            use_discounted_return = use_discounted_return,
            gamma = gamma,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            
            # report best or report current ?
            # report num_timesteps or elasped time ?            
            self.trial.report(self.last_mean_reward, self.eval_idx)

            if self.verbose >= 1:
                metric_type = "discounted" if self.use_discounted_return else "undiscounted"
                print(f"[Optuna] Reported {metric_type} return: {self.last_mean_reward:.2f}")

            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True

