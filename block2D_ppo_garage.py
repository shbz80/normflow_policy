import torch
import gym

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.torch.algos import PPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer

import normflow_policy
from normflow_policy.normflow_policy_garage import GaussianNormFlowPolicy

@wrap_experiment
def ppo_block2d(ctxt=None, seed=1):
    """Train PPO with InvertedDoublePendulum-v2 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    env = GymEnv(gym.make('Block2D-v0'))

    trainer = Trainer(ctxt)

    # policy = GaussianMLPPolicy(env.spec,
    #                            hidden_sizes=[64, 64],
    #                            hidden_nonlinearity=torch.tanh,
    #                            output_nonlinearity=None)
    policy = GaussianNormFlowPolicy(env.spec,
                            n_flows=2)

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    algo = PPO(env_spec=env.spec,
               policy=policy,
               value_function=value_function,
               discount=0.99,
               center_adv=False)

    trainer.setup(algo, env)
    trainer.train(n_epochs=100, batch_size=1500, plot=True)


ppo_block2d(seed=1)