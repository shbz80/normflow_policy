
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
    def __init__(self, loc, scale, lintrans, validate_args=None):
        super().__init__(torch.zeros_like(loc), scale, validate_args)
        self.lintrans = lintrans
        self.lintrans_pinv = torch.pinverse(self.lintrans)
    
    def sample(self, sample_shape=torch.Size()):
        samples = super().sample(sample_shape)
        #apply linear transform to samples
        return _batch_mv(self.lintrans, samples) + self.loc

    def log_prob(self, actions):
        #convert them back to subspace and eval log prob
        actions_sub = _batch_mv(self.lintrans_pinv, actions - self.loc)
        return super().log_prob(actions_sub).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)