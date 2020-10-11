import torch
import gym
from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.torch.algos import PPO
# from garage.torch.policies import GaussianMLPPolicy
# from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer
from normflow_policy.envs.yumipegcart import T, dA, dO
from garage.np.baselines import LinearFeatureBaseline
from normflow_policy.normflow_policy_garage import GaussianNormFlowPolicy
from akro.box import Box
from garage import EnvSpec
import numpy as np

# @wrap_experiment
@wrap_experiment(snapshot_mode='all')
def yumipeg_nfppo_garage(ctxt=None, seed=1):
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

    policy = GaussianNormFlowPolicy(env.spec,
                                    n_flows=2,
                                    hidden_dim=8,
                                    init_std=2.,
                                    K = 1.,
                                    D = 1.,
                                    jac_damping=True)
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
               lr_clip_range=0.2,
               # center_adv=False,
               )

    # resume_dir = '/home/shahbaz/Software/garage36/normflow_policy/data/local/experiment/yumipeg_nfppo_garage'
    # trainer.restore(resume_dir, from_epoch=47)
    # trainer.resume()
    trainer.setup(algo, env, n_workers=6)
    trainer.train(n_epochs=N, batch_size=T*S, plot=True, store_episodes=True)

yumipeg_nfppo_garage(seed=1)

#yumipeg_nfppo_garage_0,1
# seed=1
# env._action_space = Box(low=-5, high=5, shape=(dA,))
# env._observation_space = Box(low=-2, high=2.0, shape=(dO,))
# policy = GaussianNormFlowPolicy(env.spec,
#                                     n_flows=2,
#                                     hidden_dim=8,
#                                     init_std=1.,
#                                     K = 1.,
#                                     D = 1.,
#                                     jac_damping=True)
# inv=False
# value_function = LinearFeatureBaseline(env_spec=env.spec)
# N = 100  # number of epochs
# S = 15  # number of episodes in an epoch
# algo = PPO(env_spec=env.spec,
#            policy=policy,
#            value_function=value_function,
#            discount=0.99,
#            lr_clip_range=0.2,
#            )
# trainer.setup(algo, env, n_workers=4)
# GOAL = np.array([-1.50337106, -1.24545874,  1.21963181,  0.46298941,  2.18633697,  1.51383283,
#   0.57184653])
# INIT = np.array([-1.14, -1.21, 0.965, 0.728, 1.97, 1.49, 0.])
# T = 200
# dA = 3
# dO = 6
# dJ = 7
# D_rot = np.eye(3)*4
# kin_params_yumi = {}
# kin_params_yumi['urdf'] = '/home/shahbaz/Software/yumi_kinematics/yumikin/models/yumi_ABB_left.urdf'
# kin_params_yumi['base_link'] = 'world'
# # kin_params_yumi['end_link'] = 'left_tool0'
# kin_params_yumi['end_link'] = 'left_contact_point'
# kin_params_yumi['euler_string'] = 'sxyz'
# kin_params_yumi['goal'] = GOAL
# reward_params = {}
# reward_params['LIN_SCALE'] = 1
# reward_params['ROT_SCALE'] = 1
# reward_params['POS_SCALE'] = 1
# reward_params['VEL_SCALE'] = 1e-1
# reward_params['STATE_SCALE'] = 1
# reward_params['ACTION_SCALE'] = 1e-3
# reward_params['v'] = 2
# reward_params['w'] = 1
# reward_params['TERMINAL_STATE_SCALE'] = 10
# size="0.024"