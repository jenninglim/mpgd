import torch
import torch.distributions as D
import torch.nn.functional as F

class ToyHMMProblem:
    def __init__(self,
                 theta=1.,
                 scale=1.):
        self.theta = torch.ones(1) * theta
        self.scale = torch.tensor(scale)
    
    def sample(self, n_samples, seed=None):
        if seed is not None:
            torch.random.manual_seed(seed)
        x = self.theta + torch.randn(n_samples, 1) * self.scale
        y = x + torch.randn(n_samples, 1) 
        return y
         
class ToyGMM:
    def __init__(self, means=[-5., -2.5, 0., 2.5, 5.], scale=0.5):
        scale= 0.5
        dim = 1
        mix = D.Categorical(torch.ones(len(means),))
        comp = D.Independent(D.Normal(torch.tensor([[mean] for mean in means]),
                                     torch.tensor([[scale] for _ in means])), 1)
        self.gmm = D.MixtureSameFamily(mix, comp)
    
    def sample(self, n_samples, seed=None):
        if seed is not None:
            torch.random.manual_seed(seed)
        samples = self.gmm.sample([n_samples])
        return samples
