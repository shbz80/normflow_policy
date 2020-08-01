from gym.envs.registration import register

register(
    id='UJICharHandwriting-v0',
    entry_point='normflow_policy.envs:UJICharHandWritingEnv',
    max_episode_steps=100,
)

register(
     id='Block2D-v0',
     entry_point='normflow_policy.envs:Block2DEnv',
     max_episode_steps=1000,
)