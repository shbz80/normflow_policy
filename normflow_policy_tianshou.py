import numpy as np
import torch
import torch.nn as nn

from typing import Dict, List, Tuple, Union, Optional

from tianshou.data import Batch, to_torch
from tianshou.policy import PPOPolicy

from normflow_policy.normflow_ds import NormalizingFlowDynamicalSystem

class NormalizingFlowDynamicalSystemActor(nn.Module):
    """
    deterministic actor accounting stability of samples for tianshou
    """
    def __init__(self, preprocess_net, action_shape,
                 max_action, device='cpu', unbounded=False):
        super().__init__()
        self.normflow_ds = preprocess_net    #this is the normflow dynamics...
        self.normflow_ds.init_phi()
        self.device = device
        self._max = max_action
        self._unbounded = unbounded

    def forward(self, s, state=None, **kwargs):
        s = to_torch(s, device=self.device, dtype=torch.float32)
        s = s.flatten(1)
        
        x = s[:, :self.normflow_ds.dim]
        x.requires_grad_()
        x_dot = s[:, self.normflow_ds.dim:]
        x_dot.requires_grad_()

        #we need to enable grad computation because tianshou collector called no_grad with an expectation
        #of inference only torch module for policy evaluation. however, we need this for jacobian computation
        with torch.enable_grad():
            #the default destination is origin in R^n
            act = self.normflow_ds.forward_with_damping(x, torch.zeros_like(x), x_dot, inv=False, jac_damping=False)

        return act, None

class NormalizingFlowDynamicalSystemActorProb(nn.Module):
    """
    actor accounting stability of samples for tianshou
    """

    def __init__(self, preprocess_net, action_shape,
                 max_action, device='cpu', unbounded=False):
        super().__init__()
        self.normflow_ds = preprocess_net    #this is the normflow dynamics...
        self.normflow_ds.init_phi()
        self.device = device
        self._max = max_action
        self._unbounded = unbounded

        self.sigma = nn.Parameter(torch.zeros(np.prod(action_shape), 1))

    def forward(self, s, state=None, **kwargs):

        s = to_torch(s, device=self.device, dtype=torch.float32)
        s = s.flatten(1)
        
        x = s[:, :self.normflow_ds.dim]
        x.requires_grad_()
        x_dot = s[:, self.normflow_ds.dim:]
        x_dot.requires_grad_()

        #we need to enable grad computation because tianshou collector called no_grad with an expectation
        #of inference only torch module for policy evaluation. however, we need this for jacobian computation
        with torch.enable_grad():
            #the default destination is origin in R^n
            mu = self.normflow_ds.forward_with_damping(x, torch.zeros_like(x), x_dot, inv=False, jac_damping=False)

        shape = [1] * len(mu.shape)
        shape[1] = -1
        sigma = (self.sigma.view(shape) + torch.zeros_like(mu)).exp()
        return (mu, sigma), None

from normflow_policy.utils import LinearSubSpaceDiagGaussian
from tianshou.policy.dist import DiagGaussian

class NormalizingFlowDynamicalSystemPPO(PPOPolicy):
    def __init__(self,
                 actor: NormalizingFlowDynamicalSystemActorProb,
                 critic: torch.nn.Module,
                 optim: torch.optim.Optimizer,
                #  dist_fn: torch.distributions.Distribution,
                 discount_factor: float = 0.99,
                 max_grad_norm: Optional[float] = None,
                 eps_clip: float = .2,
                 vf_coef: float = .5,
                 ent_coef: float = .01,
                 action_range: Optional[Tuple[float, float]] = None,
                 gae_lambda: float = 0.95,
                 dual_clip: Optional[float] = None,
                 value_clip: bool = True,
                 reward_normalization: bool = True,
                 **kwargs) -> None:
        super().__init__(actor, critic, optim, LinearSubSpaceDiagGaussian, discount_factor, max_grad_norm, 
                eps_clip, vf_coef, ent_coef, action_range,
                gae_lambda, dual_clip, value_clip, reward_normalization, **kwargs)
    
    def forward(self, batch: Batch,
                state: Optional[Union[dict, Batch, np.ndarray]] = None,
                **kwargs) -> Batch:
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which has 4 keys:

            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``dist`` the action distribution.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        (mu, sigma), h = self.actor(batch.obs, state=state, info=batch.info)
        U = None

        #comment these out to use regular diagonal gaussian exploration
        # batch_x_dot = to_torch(batch.obs[:, self.actor.normflow_ds.dim:], device=self.actor.device, dtype=torch.float32)
        # nullspace_mat = self.actor.normflow_ds.null_space(batch_x_dot)
        # U, _, _ = torch.pca_lowrank(nullspace_mat, q=self.actor.normflow_ds.dim-1)
        
        dist = self.dist_fn(loc=mu, scale=sigma, lintrans=U)

        act = dist.sample()

        if self._range:
            act = act.clamp(self._range[0], self._range[1])
        return Batch(logits=(mu, sigma), act=act, state=h, dist=dist)





