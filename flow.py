#from https://github.com/tonyduan/normalizing-flows/blob/master/nf/flows.py
import torch
import torch.nn as nn

class FCNN(nn.Module):
    """
    Simple fully connected neural network.
    """
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.network(x)


class RealNVP(nn.Module):
    """
    Non-volume preserving flow.
    [Dinh et. al. 2017]
    """
    def __init__(self, dim, hidden_dim = 8, base_network=FCNN):
        super().__init__()
        self.D = dim
        self.Dlo = dim//2
        self.Dhi = dim - dim//2
        # self.t1 = base_network(dim // 2, dim // 2, hidden_dim)
        # self.s1 = base_network(dim // 2, dim // 2, hidden_dim)
        # self.t2 = base_network(dim // 2, dim // 2, hidden_dim)
        # self.s2 = base_network(dim // 2, dim // 2, hidden_dim)

        self.t1 = base_network(self.Dlo, self.Dhi, hidden_dim)
        self.s1 = base_network(self.Dlo, self.Dhi, hidden_dim)
        self.t2 = base_network(self.Dhi, self.Dlo, hidden_dim)
        self.s2 = base_network(self.Dhi, self.Dlo, hidden_dim)

    def forward(self, x):
        lower, upper = x[:,:self.Dlo], x[:,self.Dlo:] # todo
        # lower, upper = x[:, :self.dim//2], x[:, self.dim//2:]
        t1_transformed = self.t1(lower)
        s1_transformed = self.s1(lower)
        upper = t1_transformed + upper * torch.exp(s1_transformed)
        t2_transformed = self.t2(upper)
        s2_transformed = self.s2(upper)
        lower = t2_transformed + lower * torch.exp(s2_transformed)
        z = torch.cat([lower, upper], dim=1)
        # log_det = torch.sum(s1_transformed, dim=1) + \
        #           torch.sum(s2_transformed, dim=1)
        # return z, log_det
        return z

    def inverse(self, z):
        lower, upper = z[:,:self.D // 2], z[:,self.D // 2:]
        t2_transformed = self.t2(upper)
        s2_transformed = self.s2(upper)
        lower = (lower - t2_transformed) * torch.exp(-s2_transformed)
        t1_transformed = self.t1(lower)
        s1_transformed = self.s1(lower)
        upper = (upper - t1_transformed) * torch.exp(-s1_transformed)
        x = torch.cat([lower, upper], dim=1)
        # log_det = torch.sum(-s1_transformed, dim=1) + \
        #           torch.sum(-s2_transformed, dim=1)
        # return x, log_det
        return x
