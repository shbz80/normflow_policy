import torch
import gym
from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.torch.algos import PPO
from garage.torch.policies import GaussianMLPPolicy
# from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer
from normflow_policy.envs.yumipegcart import T, dA, dO
from garage.np.baselines import LinearFeatureBaseline
# from normflow_policy.normflow_policy_garage import GaussianNormFlowPolicy
from akro.box import Box
from garage import EnvSpec
import numpy as np

# @wrap_experiment
@wrap_experiment(snapshot_mode='all')
def yumipeg_ppo_garage(ctxt=None, seed=1):
    """Train PPO with InvertedDoublePendulum-v2 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    env = GymEnv('YumiPegCart-v0', max_episode_length=T)
    env._action_space = Box(low=-10, high=10, shape=(dA,))
    env._observation_space = Box(low=-2, high=2.0, shape=(dO,))
    env._spec = EnvSpec(action_space=env.action_space,
                             observation_space=env.observation_space,
                             max_episode_length=T)

    trainer = Trainer(ctxt)

    # policy = GaussianNormFlowPolicy(env.spec,
    #                                 n_flows=2,
    #                                 hidden_dim=16,
    #                                 init_std=3.,
    #                                 K = 1.,
    #                                 D = 1.,
    #                                 jac_damping=True)

    policy = GaussianMLPPolicy(env.spec,
                               hidden_sizes=[32, 32],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None,
                               init_std=3.)

    value_function = LinearFeatureBaseline(env_spec=env.spec)
    # value_function = GaussianMLPValueFunction(env_spec=env.spec,
    #                                           hidden_sizes=(32, 32),
    #                                           hidden_nonlinearity=torch.tanh,
    #                                           output_nonlinearity=None)
    N = 100  # number of epochs
    S = 15  # number of episodes in an epoch
    algo = PPO(env_spec=env.spec,
               policy=policy,
               value_function=value_function,
               discount=0.99,
               lr_clip_range=0.1,
               # center_adv=False,
               )

    # resume_dir = '/home/shahbaz/Software/garage36/normflow_policy/data/local/experiment/yumipeg_ppo_garage_2'
    # trainer.restore(resume_dir, from_epoch=49)
    # trainer.resume(n_epochs=100)
    trainer.setup(algo, env, n_workers=6)
    trainer.train(n_epochs=N, batch_size=T*S, plot=True, store_episodes=True)

yumipeg_ppo_garage(seed=1)
