import numpy as np
import torch
import gym
from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.torch.algos import PPO
from garage.trainer import Trainer
from normflow_policy.envs.yumipegcart import T, dA, dO
from garage.np.baselines import LinearFeatureBaseline
from normflow_policy.normflow_policy_garage import GaussianNormFlowPolicy
from akro.box import Box
from garage import EnvSpec
from garage.sampler import RaySampler, LocalSampler
from gps.agent.ros.agent_ros import AgentROS
from normflow_policy.agent_hyperparams import agent as agent_params

T = agent_params['T']
# @wrap_experiment
@wrap_experiment(snapshot_mode='all')
def yumipeg_nfppo_ros(ctxt=None, seed=1):
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
                                    n_flows=1,
                                    hidden_dim=16,
                                    init_std=5.,
                                    K = 4.,
                                    D = 1.,
                                    jac_damping=True)
    value_function = LinearFeatureBaseline(env_spec=env.spec)
    # value_function = GaussianMLPValueFunction(env_spec=env.spec,
    #                                           hidden_sizes=(32, 32),
    #                                           hidden_nonlinearity=torch.tanh,
    #                                           output_nonlinearity=None)

    N = agent_params['epoch_num']  # number of epochs
    S = agent_params['episode_num']  # number of episodes in an epoch
    algo = PPO(env_spec=env.spec,
               policy=policy,
               value_function=value_function,
               discount=0.99,
               lr_clip_range=0.2,
               # center_adv=False,
               )

    resume_dir = '/home/shahbaz/Software/garage36/normflow_policy/data/local/experiment/yumipeg_nfppo_ros_8'
    trainer.restore(resume_dir, from_epoch=48)
    trainer.resume(n_epochs=51)
    # trainer.setup(algo, env, sampler_cls=AgentROS, sampler_args= agent_params)
    # trainer.train(n_epochs=N, batch_size=T*S, plot=False, store_episodes=True)

yumipeg_nfppo_ros(seed=1)

# yumipeg_nfppo_ros(seed=1)
# GOAL = np.array([-1.5478, -1.122,  1.2577,  0.2397,  2.0677,  1.4049, -2.4206])
# ja_x0 = np.array([-1.3573, -0.8344, 1.1785, 0.4227, 1.8178, 1.2853, -2.4684]) #exp_peg_1
# T =200
# reward_params['LIN_SCALE'] = 1
# reward_params['ROT_SCALE'] = 1
# reward_params['POS_SCALE'] = 1
# reward_params['VEL_SCALE'] = 1e-1
# reward_params['STATE_SCALE'] = 1
# reward_params['ACTION_SCALE'] = 1e-3
# reward_params['v'] = 2
# reward_params['w'] = 1
# reward_params['TERMINAL_STATE_SCALE'] = 20
# reward_params['T'] = T
# agent = {
#     'dt': 0.025,
#     'conditions': common['conditions'],
#     'T': T,
#     'trial_timeout': 7,
#     'reset_timeout': 20,
#     'episode_num': 15,
#     'epoch_num': 50,
#     'x0': x0s,
#     'ee_points_tgt': ee_tgts,
#     'reset_conditions': reset_conditions,
#     'sensor_dims': SENSOR_DIMS,
#     'state_include': [JOINT_ANGLES, JOINT_VELOCITIES],
#     'end_effector_points': EE_POINTS,
#     'obs_include': [],
#     'reward': reward_params,
#     'kin_params': kin_params_yumi,
#     'K_tra': 100*np.eye(3),
#     'K_rot': 3.*np.eye(3),
# }
# env = GymEnv('YumiPegCart-v0', max_episode_length=T)
# env._action_space = Box(low=-10, high=10, shape=(dA,))
# env._observation_space = Box(low=-2, high=2.0, shape=(dO,))
# policy = GaussianNormFlowPolicy(env.spec,
#                                 n_flows=1,
#                                 hidden_dim=16,
#                                 init_std=3.,
#                                 K = 4.,
#                                 D = 1.,
#                                 jac_damping=True)
# algo = PPO(env_spec=env.spec,
#            policy=policy,
#            value_function=value_function,
#            discount=0.99,
#            lr_clip_range=0.2,
#            # center_adv=False,
#            )

# yumipeg_nfppo_ros_1,2,3,4,5,6,7,8(seed=1)
# GOAL = np.array([-1.5478, -1.122,  1.2577,  0.2397,  2.0677,  1.4049, -2.4206])
# ja_x0 = np.array([-1.3573, -0.8344, 1.1785, 0.4227, 1.8178, 1.2853, -2.4684]) #exp_peg_1
# T =200
# reward_params['LIN_SCALE'] = 1
# reward_params['ROT_SCALE'] = 0
# reward_params['POS_SCALE'] = 1
# reward_params['VEL_SCALE'] = 1e-2
# reward_params['STATE_SCALE'] = 1
# reward_params['ACTION_SCALE'] = 1e-3
# reward_params['v'] = 2
# reward_params['w'] = 1
# reward_params['TERMINAL_STATE_SCALE'] = 500
# reward_params['T'] = T
# agent = {
#     'dt': 0.025,
#     'conditions': common['conditions'],
#     'T': T,
#     'trial_timeout': 7,
#     'reset_timeout': 20,
#     'episode_num': 15,
#     'epoch_num': 50,
#     'x0': x0s,
#     'ee_points_tgt': ee_tgts,
#     'reset_conditions': reset_conditions,
#     'sensor_dims': SENSOR_DIMS,
#     'state_include': [JOINT_ANGLES, JOINT_VELOCITIES],
#     'end_effector_points': EE_POINTS,
#     'obs_include': [],
#     'reward': reward_params,
#     'kin_params': kin_params_yumi,
#     'K_tra': 100*np.eye(3),
#     'K_rot': 3.*np.eye(3),
# }
# env = GymEnv('YumiPegCart-v0', max_episode_length=T)
# env._action_space = Box(low=-10, high=10, shape=(dA,))
# env._observation_space = Box(low=-2, high=2.0, shape=(dO,))
# policy = GaussianNormFlowPolicy(env.spec,
#                                 n_flows=1,
#                                 hidden_dim=16,
#                                 init_std=5.,
#                                 K = 4.,
#                                 D = 1.,
#                                 jac_damping=True)
# algo = PPO(env_spec=env.spec,
#            policy=policy,
#            value_function=value_function,
#            discount=0.99,
#            lr_clip_range=0.2,
#            # center_adv=False,
#            )
# 9 itr 0