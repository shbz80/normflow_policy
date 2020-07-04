import torch
import torch.nn as nn
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

#batch version jacobian
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
                                                                                                      

class NormalizingFlowPolicy(nn.Module):
    
    def __init__(self, dim=2, n_flows=3):
        super().__init__()
        self.flows = [flow.RealNVP(dim, hidden_dim=8, base_network=flow.FCNN) for i in range(n_flows)]
        self.phi = nn.Sequential(*self.flows)
        self.potential = QuadraticPotentialFunction
        self.dim = dim
    
    def forward(self, x, x_star):
        phi_jac = jacobian(self.phi(x), x, create_graph=True)
        potential_grad = -self.potential.forward_grad_feature(x, x_star)
        return torch.solve(potential_grad, phi_jac)
    
    
    
