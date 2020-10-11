#!/usr/bin/env python3
"""This is an example to train a task with PPO algorithm.

Here it creates InvertedDoublePendulum using gym. And uses a PPO with 1M
steps.

Results:
    AverageDiscountedReturn: 500
    RiseTime: itr 40

"""
import tensorflow as tf

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment.deterministic import set_seed
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.policies import GaussianMLPPolicy
from garage.trainer import TFTrainer
from normflow_policy.envs.block2D import T
# from garage.tf.envs import TfEnv

@wrap_experiment(snapshot_mode='all')
# @wrap_experiment
def block2D_ppo_tf_garage(ctxt=None, seed=1):
    """Train PPO with InvertedDoublePendulum-v2 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    with TFTrainer(snapshot_config=ctxt) as trainer:
        env = GymEnv('Block2D-v0',max_episode_length=T)

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
            init_std=2,
        )

        # baseline = GaussianMLPBaseline(
        #     env_spec=env.spec,
        #     hidden_sizes=(32, 32),
        #     use_trust_region=True,
        # )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        # NOTE: make sure when setting entropy_method to 'max', set
        # center_adv to False and turn off policy gradient. See
        # tf.algos.NPO for detailed documentation.
        N = 50  # number of epochs
        S = 30  # number of episodes in an epoch
        algo = PPO(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            discount=0.99,
            gae_lambda=0.95,
            lr_clip_range=0.2,
            # optimizer_args=dict(
            #     batch_size=32,
            #     max_optimization_epochs=10,
            # ),
            # stop_entropy_gradient=True,
            # entropy_method='max',
            # policy_ent_coeff=0.02,
            # center_adv=False,
        )

        trainer.setup(algo, env, n_workers=4)

        trainer.train(n_epochs=N, batch_size=S*T, plot=True, store_episodes=True)


block2D_ppo_tf_garage(seed=1)
