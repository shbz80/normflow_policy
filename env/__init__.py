from gym.envs.registration import registry, register, make, spec

register(
    id='UJICharHandwriting-v0',
    entry_point='normflow_policy.env.ujichar_handwriting',
    max_episode_steps=200,
    reward_threshold=100.0,
)
