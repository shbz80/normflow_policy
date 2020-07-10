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
    
    def __init__(self, dim=2, n_flows=3):
        super().__init__()
        self.flows = [flow.RealNVP(dim, hidden_dim=8, base_network=flow.FCNN) for i in range(n_flows)]
        self.phi = nn.Sequential(*self.flows)
        self.potential = QuadraticPotentialFunction(feature=self.phi)
        self.dim = dim
    
    def forward(self, x, x_star, inv=False):
        '''
        x:          state pos
        x_star:     equilibrium pos
        inv:        use inverse of Jacobian or not. works as change of coordinate if True
        '''
        phi_jac = jacobian_in_batch(self.phi(x), x)
        potential_grad = -self.potential.forward_grad_feature(x, x_star).unsqueeze(-1)
        if inv:
            return torch.solve(potential_grad, phi_jac)[0].squeeze(-1)
        else:
            return torch.bmm(phi_jac.transpose(1, 2), potential_grad).squeeze(-1)

    def init_phi(self):

        def param_init(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        
        self.phi.apply(param_init)
        return
    
