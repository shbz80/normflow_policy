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
                               init_std=2.)

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

    # resume_dir = '/home/shahbaz/Software/garage36/normflow_policy/data/local/experiment/yumipeg_ppo_garage_31'
    # trainer.restore(resume_dir, from_epoch=98)
    # trainer.resume(n_epochs=100)
    trainer.setup(algo, env, n_workers=6)
    trainer.train(n_epochs=N, batch_size=T*S, plot=False, store_episodes=True)

yumipeg_ppo_garage(seed=2)

# env._observation_space = Box(low=-2, high=2.0, shape=(dO,))
# inv=False
# N = 100  # number of epochs
# S = 15  # number of episodes in an epoch
# value_function = LinearFeatureBaseline(env_spec=env.spec)
# algo = PPO(env_spec=env.spec,
#            policy=policy,
#            value_function=value_function,
#            discount=0.99,
#            lr_clip_range=0.2,
#            )
# trainer.setup(algo, env, n_workers=6)
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
# reward_params['TERMINAL_STATE_SCALE'] = 20

#yumipeg_ppo_garage
# seed=1
# env._action_space = Box(low=-10, high=10, shape=(dA,))
# policy = GaussianMLPPolicy(env.spec,
#                            hidden_sizes=[32, 32],
#                            hidden_nonlinearity=torch.tanh,
#                            output_nonlinearity=None,
#                            init_std=3.)
# reward_params['TERMINAL_STATE_SCALE'] = 20
# SIGMA = np.array([0.05,0.05,0.01])
# size="0.0235"

#yumipeg_ppo_garage_1
# seed=1
# env._action_space = Box(low=-10, high=10, shape=(dA,))
# policy = GaussianMLPPolicy(env.spec,
#                            hidden_sizes=[32, 32],
#                            hidden_nonlinearity=torch.tanh,
#                            output_nonlinearity=None,
#                            init_std=3.)
# algo = PPO(env_spec=env.spec,
#            policy=policy,
#            value_function=value_function,
#            discount=0.99,
#            lr_clip_range=0.1,
#            )
# reward_params['TERMINAL_STATE_SCALE'] = 20
# SIGMA = np.array([0.05,0.05,0.01])
# size="0.0235"

#yumipeg_ppo_garage_2,3
# seed=1
# env._action_space = Box(low=-10, high=10, shape=(dA,))
# policy = GaussianMLPPolicy(env.spec,
#                            hidden_sizes=[32, 32],
#                            hidden_nonlinearity=torch.tanh,
#                            output_nonlinearity=None,
#                            init_std=3.)
# algo = PPO(env_spec=env.spec,
#            policy=policy,
#            value_function=value_function,
#            discount=0.99,
#            lr_clip_range=0.1,
#            )
# reward_params['TERMINAL_STATE_SCALE'] = 20
# SIGMA = np.array([0.05,0.05,0.01])
# size="0.0245"

#yumipeg_ppo_garage_4
# seed=1
# env._action_space = Box(low=-10, high=10, shape=(dA,))
# policy = GaussianMLPPolicy(env.spec,
#                            hidden_sizes=[32, 32],
#                            hidden_nonlinearity=torch.tanh,
#                            output_nonlinearity=None,
#                            init_std=3.)
# algo = PPO(env_spec=env.spec,
#            policy=policy,
#            value_function=value_function,
#            discount=0.99,
#            lr_clip_range=0.1,
#            )
# reward_params['TERMINAL_STATE_SCALE'] = 20
# size="0.0245"

#yumipeg_ppo_garage_5,6
# seed=1
# env._action_space = Box(low=-10, high=10, shape=(dA,))
# policy = GaussianMLPPolicy(env.spec,
#                            hidden_sizes=[32, 32],
#                            hidden_nonlinearity=torch.tanh,
#                            output_nonlinearity=None,
#                            init_std=3.)
# algo = PPO(env_spec=env.spec,
#            policy=policy,
#            value_function=value_function,
#            discount=0.99,
#            lr_clip_range=0.05,
#            )
# reward_params['TERMINAL_STATE_SCALE'] = 20
# SIGMA_JT = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
# size="0.0235"

#yumipeg_ppo_garage_7,8
# seed=1
# env._action_space = Box(low=-10, high=10, shape=(dA,))
# policy = GaussianMLPPolicy(env.spec,
#                            hidden_sizes=[32, 32],
#                            hidden_nonlinearity=torch.tanh,
#                            output_nonlinearity=None,
#                            init_std=2.)
# algo = PPO(env_spec=env.spec,
#            policy=policy,
#            value_function=value_function,
#            discount=0.99,
#            lr_clip_range=0.1,
#            )
# reward_params['TERMINAL_STATE_SCALE'] = 20
# SIGMA_JT = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])*2
# size="0.024"

#yumipeg_ppo_garage_9
# seed=1
# env._action_space = Box(low=-10, high=10, shape=(dA,))
# policy = GaussianMLPPolicy(env.spec,
#                            hidden_sizes=[32, 32],
#                            hidden_nonlinearity=torch.tanh,
#                            output_nonlinearity=None,
#                            init_std=3.)
# algo = PPO(env_spec=env.spec,
#            policy=policy,
#            value_function=value_function,
#            discount=0.99,
#            lr_clip_range=0.1,
#            )
# reward_params['TERMINAL_STATE_SCALE'] = 20
# SIGMA_JT = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])*2
# size="0.024"

#yumipeg_ppo_garage_10
# seed=1
# env._action_space = Box(low=-10, high=10, shape=(dA,))
# policy = GaussianMLPPolicy(env.spec,
#                            hidden_sizes=[32, 32],
#                            hidden_nonlinearity=torch.tanh,
#                            output_nonlinearity=None,
#                            init_std=3.)
# algo = PPO(env_spec=env.spec,
#            policy=policy,
#            value_function=value_function,
#            discount=0.99,
#            lr_clip_range=0.1,
#            )
# reward_params['TERMINAL_STATE_SCALE'] = 20
# size="0.024"

#yumipeg_ppo_garage_11
# seed=2
# env._action_space = Box(low=-10, high=10, shape=(dA,))
# policy = GaussianMLPPolicy(env.spec,
#                            hidden_sizes=[32, 32],
#                            hidden_nonlinearity=torch.tanh,
#                            output_nonlinearity=None,
#                            init_std=3.)
# algo = PPO(env_spec=env.spec,
#            policy=policy,
#            value_function=value_function,
#            discount=0.99,
#            lr_clip_range=0.1,
#            )
# reward_params['TERMINAL_STATE_SCALE'] = 20
# size="0.024"

#yumipeg_ppo_garage_12
# seed=2
# env._action_space = Box(low=-10, high=10, shape=(dA,))
# policy = GaussianMLPPolicy(env.spec,
#                            hidden_sizes=[32, 32],
#                            hidden_nonlinearity=torch.tanh,
#                            output_nonlinearity=None,
#                            init_std=3.)
# algo = PPO(env_spec=env.spec,
#            policy=policy,
#            value_function=value_function,
#            discount=0.99,
#            lr_clip_range=0.05,
#            )
# reward_params['TERMINAL_STATE_SCALE'] = 20
# size="0.024"

#yumipeg_ppo_garage_e
# seed=1
# env._action_space = Box(low=-10, high=10, shape=(dA,))
# policy = GaussianMLPPolicy(env.spec,
#                            hidden_sizes=[32, 32],
#                            hidden_nonlinearity=torch.tanh,
#                            output_nonlinearity=None,
#                            init_std=3.)
# algo = PPO(env_spec=env.spec,
#            policy=policy,
#            value_function=value_function,
#            discount=0.99,
#            lr_clip_range=0.1,
#            # center_adv=False,
#            )
# reward_params['TERMINAL_STATE_SCALE'] = 20
# size="0.0235"
# SIGMA_JT = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])*2.

#yumipeg_ppo_garage_e_1
# seed=1
# env._action_space = Box(low=-10, high=10, shape=(dA,))
# policy = GaussianMLPPolicy(env.spec,
#                            hidden_sizes=[32, 32],
#                            hidden_nonlinearity=torch.tanh,
#                            output_nonlinearity=None,
#                            init_std=3.)
# algo = PPO(env_spec=env.spec,
#            policy=policy,
#            value_function=value_function,
#            discount=0.99,
#            lr_clip_range=0.1,
#            # center_adv=False,
#            )
# reward_params['TERMINAL_STATE_SCALE'] = 20
# size="0.0235"

#yumipeg_ppo_garage_13
# for generating deterministic sample for actual init state
# for yumipeg_ppo_garage_e_1 (fixed init)

#yumipeg_ppo_garage_14
# for generating deterministic sample for 1 of 3 rnd init state
# for yumipeg_ppo_garage_e_1 (fixed init)

#yumipeg_ppo_garage_15
# for generating deterministic sample for 2 of 3 rnd init state
# for yumipeg_ppo_garage_e_1 (fixed init)

#yumipeg_ppo_garage_16
# for generating deterministic sample for 3 of 3 rnd init state
# for yumipeg_ppo_garage_e_1 (fixed init)

#yumipeg_ppo_garage_17
# for generating deterministic sample for actual init state
# for yumipeg_ppo_garage_e (rnd init)

#yumipeg_ppo_garage_18
# for generating deterministic sample for 1 of 3 rnd init state
# for yumipeg_ppo_garage_e (rnd init)

#yumipeg_ppo_garage_19
# for generating deterministic sample for 2 of 3 rnd init state
# for yumipeg_ppo_garage_e (rnd init)

#yumipeg_ppo_garage_20
# for generating deterministic sample for 3 of 3 rnd init state
# for yumipeg_ppo_garage_e (rnd init)

# policy = GaussianMLPPolicy(env.spec,
#                                hidden_sizes=[32, 32],
#                                hidden_nonlinearity=torch.tanh,
#                                output_nonlinearity=None,
#                                init_std=3.)
#
# N = 1  # number of epochs
# S = 1  # number of episodes in an epoch
# algo = PPO(env_spec=env.spec,
#            policy=policy,
#            value_function=value_function,
#            discount=0.99,
#            lr_clip_range=0.05,
#            # center_adv=False,
#            )
# untrained
#yumipeg_ppo_garage_21
# for generating deterministic sample for actual init state
#yumipeg_ppo_garage_22
# for generating deterministic sample for 1 of 3 rnd init state
#yumipeg_ppo_garage_23
# for generating deterministic sample for 2 of 3 rnd init state
#yumipeg_ppo_garage_24
# for generating deterministic sample for 3 of 3 rnd init state
#yumipeg_ppo_garage_25
# for generating rnd sample for actual init state
#yumipeg_ppo_garage_26
# for generating rnd sample for 1 of 3 rnd init state
#yumipeg_ppo_garage_27
# for generating rnd sample for 2 of 3 rnd init state
#yumipeg_ppo_garage_28
# for generating rnd sample for 3 of 3 rnd init state

#yumipeg_ppo_garage_29
# seed=1
# env._action_space = Box(low=-10, high=10, shape=(dA,))
# policy = GaussianMLPPolicy(env.spec,
#                            hidden_sizes=[32, 32],
#                            hidden_nonlinearity=torch.tanh,
#                            output_nonlinearity=None,
#                            init_std=1.)
# algo = PPO(env_spec=env.spec,
#            policy=policy,
#            value_function=value_function,
#            discount=0.99,
#            lr_clip_range=0.1,
#            )
# reward_params['TERMINAL_STATE_SCALE'] = 20
# size="0.023"

#yumipeg_ppo_garage_30
# seed=1
# env._action_space = Box(low=-10, high=10, shape=(dA,))
# policy = GaussianMLPPolicy(env.spec,
#                            hidden_sizes=[32, 32],
#                            hidden_nonlinearity=torch.tanh,
#                            output_nonlinearity=None,
#                            init_std=0.5)
# algo = PPO(env_spec=env.spec,
#            policy=policy,
#            value_function=value_function,
#            discount=0.99,
#            lr_clip_range=0.1,
#            )
# reward_params['TERMINAL_STATE_SCALE'] = 20
# size="0.023"

#yumipeg_ppo_garage_31
##########ACCEPTED RND INIT###############
# seed=1
# env._action_space = Box(low=-10, high=10, shape=(dA,))
# policy = GaussianMLPPolicy(env.spec,
#                            hidden_sizes=[32, 32],
#                            hidden_nonlinearity=torch.tanh,
#                            output_nonlinearity=None,
#                            init_std=1.)
# algo = PPO(env_spec=env.spec,
#            policy=policy,
#            value_function=value_function,
#            discount=0.99,
#            lr_clip_range=0.1,
#            )
# reward_params['TERMINAL_STATE_SCALE'] = 20
# size="0.023"
# SIGMA_JT = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])*2.

#yumipeg_ppo_garage_32
# seed=1
# env._action_space = Box(low=-10, high=10, shape=(dA,))
# policy = GaussianMLPPolicy(env.spec,
#                            hidden_sizes=[32, 32],
#                            hidden_nonlinearity=torch.tanh,
#                            output_nonlinearity=None,
#                            init_std=1.5)
# algo = PPO(env_spec=env.spec,
#            policy=policy,
#            value_function=value_function,
#            discount=0.99,
#            lr_clip_range=0.1,
#            )
# reward_params['TERMINAL_STATE_SCALE'] = 20
# size="0.023"
# SIGMA_JT = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])*2.

#yumipeg_ppo_garage_33
# seed=1
# env._action_space = Box(low=-10, high=10, shape=(dA,))
# policy = GaussianMLPPolicy(env.spec,
#                            hidden_sizes=[32, 32],
#                            hidden_nonlinearity=torch.tanh,
#                            output_nonlinearity=None,
#                            init_std=2.)
# algo = PPO(env_spec=env.spec,
#            policy=policy,
#            value_function=value_function,
#            discount=0.99,
#            lr_clip_range=0.1,
#            )
# reward_params['TERMINAL_STATE_SCALE'] = 20
# size="0.023"
# SIGMA_JT = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])*2.

#yumipeg_ppo_garage_34
# seed=1
# env._action_space = Box(low=-10, high=10, shape=(dA,))
# policy = GaussianMLPPolicy(env.spec,
#                            hidden_sizes=[32, 32],
#                            hidden_nonlinearity=torch.tanh,
#                            output_nonlinearity=None,
#                            init_std=2.5)
# algo = PPO(env_spec=env.spec,
#            policy=policy,
#            value_function=value_function,
#            discount=0.99,
#            lr_clip_range=0.1,
#            )
# reward_params['TERMINAL_STATE_SCALE'] = 20
# size="0.023"
# SIGMA_JT = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])*2.

#yumipeg_ppo_garage_35
# deterministic samples from itr 100 from  yumipeg_ppo_garage_e_1
# seed=1
# env._action_space = Box(low=-10, high=10, shape=(dA,))
# policy = GaussianMLPPolicy(env.spec,
#                            hidden_sizes=[32, 32],
#                            hidden_nonlinearity=torch.tanh,
#                            output_nonlinearity=None,
#                            init_std=3.)
# algo = PPO(env_spec=env.spec,
#            policy=policy,
#            value_function=value_function,
#            discount=0.99,
#            lr_clip_range=0.1,
#            )
# reward_params['TERMINAL_STATE_SCALE'] = 20
# size="0.0235" should have been 0.023 but it is ok
# SIGMA_JT = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])*3.

#deterministic samples from yumipeg_ppo_garage_31
# seed=1
# env._action_space = Box(low=-10, high=10, shape=(dA,))
# policy = GaussianMLPPolicy(env.spec,
#                            hidden_sizes=[32, 32],
#                            hidden_nonlinearity=torch.tanh,
#                            output_nonlinearity=None,
#                            init_std=1.)
# algo = PPO(env_spec=env.spec,
#            policy=policy,
#            value_function=value_function,
#            discount=0.99,
#            lr_clip_range=0.1,
#            )
# reward_params['TERMINAL_STATE_SCALE'] = 20
# size="0.023"
# SIGMA_JT = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])*4. (fixed list of random samples)

# #yumipeg_ppo_garage_36, itr 0, sample 0
# #yumipeg_ppo_garage_37, itr 0, sample 1
# #yumipeg_ppo_garage_38, itr 0, sample 2
# #yumipeg_ppo_garage_39, itr 0, sample 3
# #yumipeg_ppo_garage_40, itr 0, sample 4

# #yumipeg_ppo_garage_41, itr 9, sample 0
# #yumipeg_ppo_garage_42, itr 9, sample 1
# #yumipeg_ppo_garage_43, itr 9, sample 2
# #yumipeg_ppo_garage_44, itr 9, sample 3
# #yumipeg_ppo_garage_45, itr 9, sample 4

# #yumipeg_ppo_garage_46, itr 99, sample 0
# #yumipeg_ppo_garage_47, itr 99, sample 1
# #yumipeg_ppo_garage_48, itr 99, sample 2
# #yumipeg_ppo_garage_49, itr 99, sample 3
# #yumipeg_ppo_garage_50, itr 99, sample 4

#yumipeg_ppo_garage_e_3
# seed=1
# env._action_space = Box(low=-10, high=10, shape=(dA,))
# policy = GaussianMLPPolicy(env.spec,
#                            hidden_sizes=[32, 32],
#                            hidden_nonlinearity=torch.tanh,
#                            output_nonlinearity=None,
#                            init_std=3)
# algo = PPO(env_spec=env.spec,
#            policy=policy,
#            value_function=value_function,
#            discount=0.99,
#            lr_clip_range=0.1,
#            )
# reward_params['TERMINAL_STATE_SCALE'] = 20
# size="0.023"
# SIGMA_JT = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])*2.

#yumipeg_ppo_garage_51
# seed=2
# env._action_space = Box(low=-10, high=10, shape=(dA,))
# policy = GaussianMLPPolicy(env.spec,
#                            hidden_sizes=[32, 32],
#                            hidden_nonlinearity=torch.tanh,
#                            output_nonlinearity=None,
#                            init_std=2)
# algo = PPO(env_spec=env.spec,
#            policy=policy,
#            value_function=value_function,
#            discount=0.99,
#            lr_clip_range=0.1,
#            )
# reward_params['TERMINAL_STATE_SCALE'] = 20
# size="0.023"
# SIGMA_JT = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])*2.