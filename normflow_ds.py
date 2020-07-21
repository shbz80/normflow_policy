import torch
import torch.nn as nn
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
    
    def __init__(self, dim=2, n_flows=3, K=None, D=None):
        super().__init__()
        self.flows = [flow.RealNVP(dim, hidden_dim=8, base_network=flow.FCNN) for i in range(n_flows)]
        self.phi = nn.Sequential(*self.flows)
        self.potential = QuadraticPotentialFunction(feature=self.phi)
        self.dim = dim
        
        if K is None:
            self.K = torch.eye(self.dim)
        elif isinstance(K, (int, float)):
            self.K = torch.eye(self.dim) * K
        else:
            self.K = K

        if D is None:
            self.D = torch.eye(self.dim)
        elif isinstance(D, (int, float)):
            self.D = torch.eye(self.dim) * D
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
    
    def null_space_proj(self, x, plane_norm):
        '''
        project x to the plane defined by plane_norm, batch-wise processing
        x:          batch of vectors with dim length
        plane_norm: batch of norms
        '''
        norm_dir = torch.div(plane_norm, torch.clamp(torch.sum(plane_norm**2, dim=1, keepdim=True), min=1e-6))
        proj_len = torch.bmm(x.view(x.shape[0], 1, x.shape[1]), plane_norm.view(plane_norm.shape[0], plane_norm.shape[1], 1)).squeeze(-1)
        return x - proj_len*norm_dir

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
        print(x, x_dot)
        #the default destination is origin in R^n
        mu = self.normflow_ds.forward_with_damping(x, torch.zeros_like(x), x_dot).detach()

        shape = [1] * len(mu.shape)
        shape[1] = -1
        sigma = (self.sigma.view(shape) + torch.zeros_like(mu)).exp()
        return (mu, sigma), None
    
class NormalizingFlowDynamicalSystemPPO(PPOPolicy):
    def __init__(self,
                 actor: NormalizingFlowDynamicalSystemActorProb,
                 critic: torch.nn.Module,
                 optim: torch.optim.Optimizer,
                 dist_fn: torch.distributions.Distribution,
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
        super().__init__(actor, critic, optim, dist_fn, discount_factor, max_grad_norm, 
                eps_clip, vf_coef, ent_coef, action_range,
                gae_lambda, dual_clip, value_clip, reward_normalization, **kwargs)
    
    def forward(self, batch: Batch,
                state: Optional[Union[dict, Batch, np.ndarray]] = None,
                **kwargs) -> Batch:
        ret_batch = super().forward(batch, state, **kwargs)
        #project batch action to nullspace to maintain stability
        batch_x_dot = batch.obs[:, self.actor.normflow_ds.dim:]
        ret_batch.act = self.actor.normflow_ds.null_space_proj(ret_batch.act, batch_x_dot)
        return ret_batch