import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import flow

class QuadraticPotentialFunction(nn.Module):

    def __init__(self, feature=None):
        super().__init__()

        self.feature = feature
    
    def forward(self, x, x_star):
        if self.feature is not None:
            x = self.feature(x)
            x_star = self.feature(x_star)
        
        return (x - x_star).pow(2).sum(1)
    
    def forward_grad_feature(self, x, x_star):
        if self.feature is not None:
            x = self.feature(x)
            x_star = self.feature(x_star)
        
        return (x - x_star)*2

#https://gist.github.com/apaszke/226abdf867c4e9d6698bd198f3b45fb7
def jacobian(y, x, create_graph=False):                                                               
    jac = []                                                                                          
    flat_y = y.reshape(-1)                                                                            
    grad_y = torch.zeros_like(flat_y)                                                                 
    for i in range(len(flat_y)):                                                                      
        grad_y[i] = 1.                                                                                
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))                                                           
        grad_y[i] = 0.                                                                                
    return torch.stack(jac).reshape(y.shape + x.shape)


#batch version jacobian
#https://github.com/pytorch/pytorch/issues/23475
def jacobian_in_batch(y, x):
    '''
    Compute the Jacobian matrix in batch form.
    Return (B, D_y, D_x)
    '''
    batch = y.shape[0]
    single_y_size = np.prod(y.shape[1:])
    y = y.view(batch, -1)
    vector = torch.ones(batch).to(y)

    # Compute Jacobian row by row.
    # dy_i / dx -> dy / dx
    # (B, D) -> (B, 1, D) -> (B, D, D)
    jac = [torch.autograd.grad(y[:, i], x, 
                               grad_outputs=vector, 
                               retain_graph=True,
                               create_graph=True)[0].view(batch, -1)
                for i in range(single_y_size)]
    jac = torch.stack(jac, dim=1)
    
    return jac                                                
                                                                                                      

class NormalizingFlowDynamicalSystem(nn.Module):
    
    def __init__(self, dim=2, n_flows=3, hidden_dim=8, K=None, D=None, device='cpu'):
        super().__init__()
        self.flows = [flow.RealNVP(dim, hidden_dim=hidden_dim, base_network=flow.FCNN) for i in range(n_flows)]
        self.phi = nn.Sequential(*self.flows)
        self.potential = QuadraticPotentialFunction(feature=self.phi)
        self.dim = dim
        self.device = device

        if device == 'cpu':
            self.phi.cpu()
            self.potential.cpu()
        else:
            self.phi.cuda()
            self.potential.cuda()
        
        if K is None:
            self.K = torch.eye(self.dim, device=device)
        elif isinstance(K, (int, float)):
            self.K = torch.eye(self.dim, device=device) * K
        else:
            self.K = K

        if D is None:
            self.D = torch.eye(self.dim, device=device)
        elif isinstance(D, (int, float)):
            self.D = torch.eye(self.dim, device=device) * D
        else:
            self.D = D
    
    def forward(self, x, x_star, inv=False):
        '''
        x:          state pos
        x_star:     equilibrium pos
        inv:        use inverse of Jacobian or not. works as change of coordinate if True
        '''
        y = self.phi(x)
        phi_jac = jacobian_in_batch(y, x)
        potential_grad = -self.potential.forward_grad_feature(x, x_star).unsqueeze(-1)
        if inv:
            return torch.solve(potential_grad, phi_jac)[0].squeeze(-1)
        else:
            return torch.bmm(phi_jac.transpose(1, 2), potential_grad).squeeze(-1)
    
    def forward_with_damping(self, x, x_star, x_dot, inv=False, jac_damping=True):
        '''
        same as forward
        D:              damping matrix
        x_dot:          time derivative of x
        jac_damping:    apply jacobian to damping matrix?
        '''
        y = self.phi(x)
        # print(y.requires_grad, x.requires_grad)
        phi_jac = jacobian_in_batch(y, x)
        potential_grad = -self.potential.forward_grad_feature(x, x_star).unsqueeze(-1)

        if jac_damping:
            damping_acc = -torch.bmm(
                torch.bmm(
                    torch.bmm(phi_jac.transpose(1, 2), self.D.expand(x_dot.shape[0], -1, -1)), 
                    phi_jac), 
                x_dot.unsqueeze(-1)).squeeze(-1)
        else:
            damping_acc = -torch.bmm(self.D.expand(x_dot.shape[0], -1, -1), x_dot.unsqueeze(-1)).squeeze(-1)

        if inv:
            return torch.solve(potential_grad, phi_jac)[0].squeeze(-1) + damping_acc
        else: 
            return torch.bmm(phi_jac.transpose(1, 2), potential_grad).squeeze(-1) + damping_acc
    
    def potential_with_damping(self, x, x_star, x_dot, M):
        #M: batched version of mass, could be spd depending on x
        x_potential = 0.5*self.potential.forward(x, x_star)
        x_dot_potential = 0.5*torch.bmm(torch.bmm(x_dot.unsqueeze(1), M), x_dot).squeeze(-1)
        return x_potential + x_dot_potential

    def null_space_proj(self, x, plane_norm):
        '''
        project x to the plane defined by plane_norm, batch-wise processing
        x:          batch of vectors with dim length
        plane_norm: batch of norms
        '''
        norm_dir = F.normalize(plane_norm, dim=1)
        proj_len = torch.bmm(x.view(x.shape[0], 1, x.shape[1]), norm_dir.view(norm_dir.shape[0], norm_dir.shape[1], 1)).squeeze(-1)
        return x - proj_len*norm_dir
    
    def null_space(self, x_dot):
        '''
        get nullspace of given batch of x_dot such that
        torch.bmm(nullspace, x_dot) == 0

        return (batch_size, x_dot_dim, x_dot_dim)
        '''
        #note we can avoid matrix inversion because x_dot are vectors so we actually just need the inverse of norm
        norm_square_inv = 1./torch.sum(x_dot**2, dim=1, keepdim=True).clamp(min=1e-4)
        # print('x_dot', x_dot)
        I = torch.eye(x_dot.shape[1], device=self.device).unsqueeze(0).repeat(x_dot.shape[0], 1, 1)
        return I - norm_square_inv.unsqueeze(-1)*torch.bmm(x_dot.unsqueeze(-1), x_dot.unsqueeze(1))

    def init_phi(self):

        def param_init(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        
        self.phi.apply(param_init)
        return

from typing import Dict, List, Tuple, Union, Optional

from tianshou.data import Batch, to_torch
from tianshou.policy import PPOPolicy

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
            mu = self.normflow_ds.forward_with_damping(x, torch.zeros_like(x), x_dot, inv=False, jac_damping=True)

        shape = [1] * len(mu.shape)
        shape[1] = -1
        sigma = (self.sigma.view(shape) + torch.zeros_like(mu)).exp()
        return (mu, sigma), None

from normflow_policy.utils import LinearSubSpaceDiagGaussian

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
        
        batch_x_dot = to_torch(batch.obs[:, self.actor.normflow_ds.dim:], device=self.actor.device, dtype=torch.float32)

        nullspace_mat = self.actor.normflow_ds.null_space(batch_x_dot)

        #construct conv factor, also add a small regularization term to keep it valid
        # cov = torch.bmm(torch.bmm(nullspace_mat, torch.diag_embed(sigma)), nullspace_mat.transpose(1, 2)) + torch.diag_embed(torch.ones_like(sigma))*1e-4
        
        # cov = torch.diag_embed(sigma)
        #must be multivariate gaussian        
        dist = self.dist_fn(loc=mu, scale=sigma, lintrans=nullspace_mat)

        # act = mu + torch.bmm(nullspace_mat, (torch.randn_like(mu)*sigma).unsqueeze(-1)).squeeze(-1) 
        act = dist.sample()

        if self._range:
            act = act.clamp(self._range[0], self._range[1])
        return Batch(logits=(mu, sigma), act=act, state=h, dist=dist)





