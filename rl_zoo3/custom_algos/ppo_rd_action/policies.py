# This file is here just to define MlpPolicy/CnnPolicy
# that work for PPO
from rl_zoo3.custom_algos.ppo_rd_action.policies_rd import ActorCriticCnnPolicy, ActorCriticPolicy, MultiInputActorCriticPolicy

MlpPolicy = ActorCriticPolicy
CnnPolicy = ActorCriticCnnPolicy
MultiInputPolicy = MultiInputActorCriticPolicy
