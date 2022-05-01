import torch
import gym
from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.torch.algos import PPO
# from garage.torch.policies import GaussianMLPPolicy
# from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer
from normflow_policy.envs.block2D import T
from garage.np.baselines import LinearFeatureBaseline
from normflow_policy.normflow_policy_garage import GaussianNormFlowPolicy

# @wrap_experiment
@wrap_experiment(snapshot_mode='all')
def block2d_nfppo_garage(ctxt=None, seed=1):
    """Train PPO with InvertedDoublePendulum-v2 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    env = GymEnv('Block2D-v0',max_episode_length=T)
    trainer = Trainer(ctxt)

    policy = GaussianNormFlowPolicy(env.spec,
                                    n_flows=2,
                                    hidden_dim = 8,
                                    init_std=2.,
                                    jac_damping=True)
    value_function = LinearFeatureBaseline(env_spec=env.spec)
    # value_function = GaussianMLPValueFunction(env_spec=env.spec,
    #                                           hidden_sizes=(32, 32),
    #                                           hidden_nonlinearity=torch.tanh,
    #                                           output_nonlinearity=None)
    N = 10  # number of epochs
    S = 15  # number of episodes in an epoch
    algo = PPO(env_spec=env.spec,
               policy=policy,
               value_function=value_function,
               discount=0.99,
               lr_clip_range=0.2,
               # center_adv=False,
               )
    #resume_dir = '/home/shahbaz/Software/garage36/normflow_policy/data/local/experiment/block2d_nfppo_garage_e_3'
    #trainer.restore(resume_dir, from_epoch=99)
    #trainer.resume(n_epochs=101)
    trainer.setup(algo, env, n_workers=4)
    trainer.train(n_epochs=N, batch_size=T*S, plot=True, store_episodes=True)

block2d_nfppo_garage(seed=1)

# value_function = LinearFeatureBaseline(env_spec=env.spec)
# algo = PPO(env_spec=env.spec,
#            policy=policy,
#            value_function=value_function,
#            discount=0.99,
#            lr_clip_range=0.2,
#            )
# trainer.setup(algo, env, n_workers=6)
# T = 200
# POS_SCALE = 1
# VEL_SCALE = 0.1
# ACTION_SCALE = 1e-3
# v = 2
# w = 1
# TERMINAL_STATE_SCALE = 10
# OFFSET = np.array([0.4, -0.6])
# GOAL = np.array([0, 0.5])+OFFSET

#block2d_nfppo_garage_1
# seed=1
# policy = GaussianNormFlowPolicy(env.spec,
#                                 n_flows=2,
#                                 init_std=2
#                                 jac_damping=True)
# inv=False
# value_function = LinearFeatureBaseline(env_spec=env.spec)
# N = 50  # number of epochs
# S = 15  # number of episodes in an epoch
# INIT = np.array([-0.3, 0.8])+OFFSET # pos1
# size="0.05 0.048 0.05" mass ="1"

#block2d_nfppo_garage_3
# seed=1
# policy = GaussianNormFlowPolicy(env.spec,
#                                 n_flows=2,
#                                 init_std=2
#                                 jac_damping=False)
# inv=False
# N = 50  # number of epochs
# S = 15  # number of episodes in an epoch
# INIT = np.array([-0.3, 0.8])+OFFSET # pos1
# size="0.05 0.048 0.05" mass ="1"

#block2d_nfppo_garage_4
# seed=1
# policy = GaussianNormFlowPolicy(env.spec,
#                                 n_flows=2,
#                                 init_std=2
#                                 jac_damping=True)
# inv=False
# N = 50  # number of epochs
# S = 15  # number of episodes in an epoch
# INIT = np.array([-0.3, 0.8])+OFFSET # pos1
# SIGMA = 0.05
# size="0.05 0.048 0.05" mass ="1"

#block2d_nfppo_garage_5
# seed=1
# policy = GaussianNormFlowPolicy(env.spec,
#                                 n_flows=2,
#                                 init_std=2
#                                 jac_damping=True)
# inv=False
# N = 50  # number of epochs
# S = 15  # number of episodes in an epoch
# OFFSET_1 = np.array([0, -0.1])
# INIT = np.array([-0.3, 0.8])+OFFSET+OFFSET_1 # NFPPPO
# SIGMA = np.array([0.05, 0.1])
# size="0.05 0.049 0.05" mass ="1"

#block2d_nfppo_garage_6
# seed=1
# policy = GaussianNormFlowPolicy(env.spec,
#                                 n_flows=2,
#                                 init_std=2
#                                 jac_damping=True)
# inv=False
# N = 50  # number of epochs
# S = 15  # number of episodes in an epoch
# OFFSET_1 = np.array([0, -0.1])
# INIT = np.array([-0.3, 0.8])+OFFSET+OFFSET_1 # NFPPPO
# SIGMA = np.array([0.05, 0.1])
# size="0.05 0.048 0.05" mass ="1"

#block2d_nfppo_garage_7
# seed=1
# policy = GaussianNormFlowPolicy(env.spec,
#                                 n_flows=2,
#                                 init_std=2
#                                 jac_damping=True)
# inv=False
# N = 100  # number of epochs
# S = 15  # number of episodes in an epoch
# OFFSET_1 = np.array([0, -0.1])
# INIT = np.array([-0.3, 0.8])+OFFSET+OFFSET_1 # NFPPPO
# SIGMA = np.array([0.05, 0.1])
# size="0.05 0.048 0.05" mass ="1"

#block2d_nfppo_garage_8
# same as above with seed=2, but worse
# seed=2
# policy = GaussianNormFlowPolicy(env.spec,
#                                 n_flows=2,
#                                 init_std=2
#                                 jac_damping=True)
# inv=False
# N = 100  # number of epochs
# S = 15  # number of episodes in an epoch
# OFFSET_1 = np.array([0, -0.1])
# INIT = np.array([-0.3, 0.8])+OFFSET+OFFSET_1 # NFPPPO
# SIGMA = np.array([0.05, 0.1])
# size="0.05 0.048 0.05" mass ="1"

#block2d_nfppo_garage_9,10
# ###########accepted for fixed init###############
# seed=1
# policy = GaussianNormFlowPolicy(env.spec,
#                                 n_flows=2,
#                                 init_std=2
#                                 jac_damping=True)
# inv=False
# N = 100  # number of epochs
# S = 15  # number of episodes in an epoch
# OFFSET_1 = np.array([0, 0])
# INIT = np.array([-0.3, 0.8])+OFFSET+OFFSET_1 # NFPPPO
# SIGMA = np.array([0, 0]) Fixed
# size="0.05 0.048 0.05" mass ="1"

#block2d_nfppo_garage_11
# seed=1
# policy = GaussianNormFlowPolicy(env.spec,
#                                     n_flows=2,
#                                     hidden_dim = 16,
#                                     init_std=2,
#                                     jac_damping=True)
# algo = PPO(env_spec=env.spec,
#            policy=policy,
#            value_function=value_function,
#            discount=0.99,
#            lr_clip_range=0.1,
#            # center_adv=False,
#            )
# inv=False
# N = 100  # number of epochs
# S = 15  # number of episodes in an epoch
# OFFSET_1 = np.array([0, 0])
# INIT = np.array([-0.3, 0.8])+OFFSET+OFFSET_1 # NFPPPO
# SIGMA = np.array([0, 0]) Fixed
# size="0.05 0.048 0.05" mass ="1"

#block2d_nfppo_garage_12
# deterministic sample from init policy
# seed=1
# policy = GaussianNormFlowPolicy(env.spec,
#                                     n_flows=2,
#                                     hidden_dim = 8,
#                                     init_std=2,
#                                     jac_damping=True)
# algo = PPO(env_spec=env.spec,
#            policy=policy,
#            value_function=value_function,
#            discount=0.99,
#            lr_clip_range=0.2,
#            # center_adv=False,
#            )
# inv=False
# N = 1  # number of epochs
# S = 1  # number of episodes in an epoch
# OFFSET_1 = np.array([0, 0])
# INIT = np.array([-0.3, 0.8])+OFFSET+OFFSET_1 # NFPPPO
# SIGMA = np.array([0, 0]) Fixed
# size="0.05 0.048 0.05" mass ="1"

#block2d_nfppo_garage_13
# deterministic sample from policy itr 99 block2d_nfppo_garage_11
# it should have been from 10 but it is not working
# seed=1
# policy = GaussianNormFlowPolicy(env.spec,
#                                     n_flows=2,
#                                     hidden_dim = 8,
#                                     init_std=2,
#                                     jac_damping=True)
# algo = PPO(env_spec=env.spec,
#            policy=policy,
#            value_function=value_function,
#            discount=0.99,
#            lr_clip_range=0.2,
#            # center_adv=False,
#            )
# inv=False
# N = 1  # number of epochs
# S = 1  # number of episodes in an epoch
# OFFSET_1 = np.array([0, 0])
# INIT = np.array([-0.3, 0.8])+OFFSET+OFFSET_1 # NFPPPO
# SIGMA = np.array([0, 0]) Fixed
# size="0.05 0.048 0.05" mass ="1"

# block2d_nfppo_garage_e
# policy = GaussianNormFlowPolicy(env.spec,
#                                     n_flows=2,
#                                     hidden_dim = 8,
#                                     init_std=2,
#                                     jac_damping=True)
#     N = 100  # number of epochs
#     S = 15  # number of episodes in an epoch
#     algo = PPO(env_spec=env.spec,
#                policy=policy,
#                value_function=value_function,
#                discount=0.99,
#                lr_clip_range=0.1,
#                # center_adv=False,
#                )
#     trainer.setup(algo, env, n_workers=6)
# block2d_nfppo_garage(seed=1)
# size="0.05 0.048 0.05" mass ="1"
# SIGMA = np.array([0.05, 0.1]) # NFPPO ?

# block2d_nfppo_garage_e_1
# policy = GaussianNormFlowPolicy(env.spec,
#                                     n_flows=2,
#                                     hidden_dim = 8,
#                                     init_std=3,
#                                     jac_damping=True)
#     N = 100  # number of epochs
#     S = 15  # number of episodes in an epoch
#     algo = PPO(env_spec=env.spec,
#                policy=policy,
#                value_function=value_function,
#                discount=0.99,
#                lr_clip_range=0.1,
#                # center_adv=False,
#                )
#     trainer.setup(algo, env, n_workers=6)
# block2d_nfppo_garage(seed=1)
# size="0.05 0.048 0.05" mass ="1"
# SIGMA = np.array([0.05, 0.1]) # NFPPO ?

# block2d_nfppo_garage_e_2
# policy = GaussianNormFlowPolicy(env.spec,
#                                     n_flows=2,
#                                     hidden_dim = 8,
#                                     init_std=2,
#                                     jac_damping=True)
#     N = 100  # number of epochs
#     S = 15  # number of episodes in an epoch
#     algo = PPO(env_spec=env.spec,
#                policy=policy,
#                value_function=value_function,
#                discount=0.99,
#                lr_clip_range=0.1,
#                # center_adv=False,
#                )
#     trainer.setup(algo, env, n_workers=6)
# block2d_nfppo_garage(seed=1)
# OFFSET_1 = np.array([0, -0.1])
# SIGMA = np.array([0.05, 0.1]) # NFPPO
# size="0.05 0.048 0.05" mass ="1"

# block2d_nfppo_garage_e_3
######accepted for rnd init############
# policy = GaussianNormFlowPolicy(env.spec,
#                                     n_flows=2,
#                                     hidden_dim = 8,
#                                     init_std=2,
#                                     jac_damping=True)
#     N = 100  # number of epochs
#     S = 15  # number of episodes in an epoch
#     algo = PPO(env_spec=env.spec,
#                policy=policy,
#                value_function=value_function,
#                discount=0.99,
#                lr_clip_range=0.2,
#                # center_adv=False,
#                )
#     trainer.setup(algo, env, n_workers=6)
# block2d_nfppo_garage(seed=1)
# OFFSET_1 = np.array([0, -0.1])
# SIGMA = np.array([0.05, 0.1]) # NFPPO
# size="0.05 0.048 0.05" mass ="1"

# block2d_nfppo_garage_14
# policy = GaussianNormFlowPolicy(env.spec,
#                                     n_flows=2,
#                                     hidden_dim = 8,
#                                     init_std=0.5,
#                                     jac_damping=True)
#     N = 100  # number of epochs
#     S = 15  # number of episodes in an epoch
#     algo = PPO(env_spec=env.spec,
#                policy=policy,
#                value_function=value_function,
#                discount=0.99,
#                lr_clip_range=0.1,
#                # center_adv=False,
#                )
#     trainer.setup(algo, env, n_workers=6)
# block2d_nfppo_garage(seed=1)
# size="0.05 0.048 0.05" mass ="1"
# SIGMA = np.array([0.05, 0.1])

# block2d_nfppo_garage_15
# policy = GaussianNormFlowPolicy(env.spec,
#                                     n_flows=2,
#                                     hidden_dim = 8,
#                                     init_std=0.75,
#                                     jac_damping=True)
#     N = 100  # number of epochs
#     S = 15  # number of episodes in an epoch
#     algo = PPO(env_spec=env.spec,
#                policy=policy,
#                value_function=value_function,
#                discount=0.99,
#                lr_clip_range=0.1,
#                # center_adv=False,
#                )
#     trainer.setup(algo, env, n_workers=6)
# block2d_nfppo_garage(seed=1)
# size="0.05 0.048 0.05" mass ="1"
# SIGMA = np.array([0.05, 0.1])

# block2d_nfppo_garage_16
# policy = GaussianNormFlowPolicy(env.spec,
#                                     n_flows=2,
#                                     hidden_dim = 8,
#                                     init_std=1.,
#                                     jac_damping=True)
#     N = 100  # number of epochs
#     S = 15  # number of episodes in an epoch
#     algo = PPO(env_spec=env.spec,
#                policy=policy,
#                value_function=value_function,
#                discount=0.99,
#                lr_clip_range=0.1,
#                # center_adv=False,
#                )
#     trainer.setup(algo, env, n_workers=6)
# block2d_nfppo_garage(seed=1)
# size="0.05 0.048 0.05" mass ="1"
# SIGMA = np.array([0.05, 0.1])

# block2d_nfppo_garage_18
# policy = GaussianNormFlowPolicy(env.spec,
#                                     n_flows=2,
#                                     hidden_dim = 8,
#                                     init_std=1.,
#                                     jac_damping=True)
#     N = 100  # number of epochs
#     S = 15  # number of episodes in an epoch
#     algo = PPO(env_spec=env.spec,
#                policy=policy,
#                value_function=value_function,
#                discount=0.99,
#                lr_clip_range=0.2,
#                # center_adv=False,
#                )
#     trainer.setup(algo, env, n_workers=6)
# block2d_nfppo_garage(seed=1)
# size="0.05 0.048 0.05" mass ="1"
# SIGMA = np.array([0.05, 0.1])

# block2d_nfppo_garage_19, deterministic sample from itr0
# OFFSET_1 = np.array([0, 0])
# INIT = np.array([-0.3, 0.8])+OFFSET+OFFSET_1 # NFPPPO
# SIGMA = np.array([0, 0]) Fixed
# size="0.05 0.048 0.05" mass ="1"

# block2d_nfppo_garage_20, deterministic sample from itr100 of block2d_nfppo_garage_e_3
# OFFSET_1 = np.array([0, 0])
# INIT = np.array([-0.3, 0.8])+OFFSET+OFFSET_1 # NFPPPO
# SIGMA = np.array([0, 0]) Fixed
# size="0.05 0.048 0.05" mass ="1"