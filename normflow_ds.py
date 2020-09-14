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
        x_dot_potential = 0.5*torch.bmm(torch.bmm(x_dot.unsqueeze(1), M), x_dot.unsqueeze(-1)).squeeze()
        # print(x_potential.shape, x_dot_potential.shape)
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
        norm_square_inv = 1./torch.sum(x_dot**2, dim=1, keepdim=True).clamp(min=1e-6)
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

