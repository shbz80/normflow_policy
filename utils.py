
import torch

from tianshou.policy.dist import DiagGaussian

def _batch_mv(bmat, bvec):
    r"""
    Performs a batched matrix-vector product, with compatible but different batch shapes.

    This function takes as input `bmat`, containing :math:`n \times n` matrices, and
    `bvec`, containing length :math:`n` vectors.

    Both `bmat` and `bvec` may have any number of leading dimensions, which correspond
    to a batch shape. They are not necessarily assumed to have the same batch shape,
    just ones which can be broadcasted.
    """
    return torch.matmul(bmat, bvec.unsqueeze(-1)).squeeze(-1)

class LinearSubSpaceDiagGaussian(DiagGaussian):
    """Diagonal Gaussian distribution in specified linear subspace"""
    """This is used as a regular diag-gaussian if transformation is not given"""
    def __init__(self, loc, scale, lintrans, validate_args=None):
        if lintrans is not None:
            scale = scale[:, :-1]
        super().__init__(torch.zeros_like(scale), scale, validate_args)
        
        self.loc_real = loc
        self.lintrans = lintrans
        if lintrans is not None:
            #self.lintrans_pinv = torch.pinverse(self.lintrans)
            self.lintrans_pinv = lintrans.transpose(-2, -1)
        else:
            self.lintrans_pinv = None

    def sample(self, sample_shape=torch.Size()):
        samples = super().sample(sample_shape)
        if self.lintrans is not None:
            #apply linear transform to samples
            # print(self.lintrans.shape, samples.shape)
            return _batch_mv(self.lintrans, samples) + self.loc_real
        else:
            #linear subspace is void, we are free to generate samples
            return samples + self.loc_real

    def log_prob(self, actions):
        #convert them back to subspace and eval log prob
        if self.lintrans is not None:
            #convert them back to subspace and eval log prob
            actions_sub = _batch_mv(self.lintrans_pinv, actions - self.loc_real)
        else:
            actions_sub = actions - self.loc_real
        return super().log_prob(actions_sub).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy()
    

if __name__ == '__main__':
    batch_size=4
    loc = torch.randn((batch_size, 5))
    scale = torch.rand((batch_size, 5)) + 1
    diag_dist = DiagGaussian(loc=loc, scale=scale)
    linsub_dist = LinearSubSpaceDiagGaussian(loc=loc, scale=scale, lintrans=None)

    ## are they identical?
    samples = diag_dist.sample()
    print(diag_dist.log_prob(samples), linsub_dist.log_prob(samples))
    samples = linsub_dist.sample()
    print(diag_dist.log_prob(samples), linsub_dist.log_prob(samples))