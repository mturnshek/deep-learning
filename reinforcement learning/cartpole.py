from dqn import learning
from dqn import models

# can test longer max episode duration with this
# gym.envs.register(
#     id='CartPole-v2',
#     entry_point='gym.envs.classic_control:CartPoleEnv',
#     tags={'wrapper_config.TimeLimit.max_episode_steps': 100000},
#     reward_threshold=195.0,
# )

agent = learning.Agent('CartPole-v1', models.simple)
agent.play()