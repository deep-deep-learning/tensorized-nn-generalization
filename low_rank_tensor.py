import tltorch
import torch
import torch.nn as nn
from torch.distributions.half_cauchy import HalfCauchy
from torch.distributions.normal import Normal
from .distribution import LogUniform
import numpy as np

class LowRankTensor(nn.Module):

    def __init__(self, in_features, out_features, prior_type='log_uniform', eta=None, device=None, dtype=None):
        '''
        arguments:
            in_features: an int, the input size
            out_features: an int, the output size
            prior_type: a str, the prior distribution of the rank parameter:
                        'log_uniform' or 'half_cauchy'
            eta: a float, the hyper-parameter if prior_type == 'half_cauchy'
        '''

        super().__init__()

        # initialize rank parameter prior
        if prior_type == 'half_cauchy':
            self.rank_parameter_prior_distribution = HalfCauchy(eta)
        elif prior_type == 'log_uniform':
            self.rank_parameter_prior_distribution = LogUniform(torch.tensor([1e-30], device=device, dtype=dtype), 
                                                                torch.tensor([1e30], device=device, dtype=dtype))

        # get tensorized shape: (in_features, out_fetures) -> ((tensorized_in), (tensorized_out))
        # where tensorized_in is a tuple of ints, where tensorized_in[i] is the i-th mode input dimension
        # and tensorized_out is a tuple of ints, where tensorized_out[i] is the i-th mode output dimension
        # ex) (128, 256) -> ((2, 4, 8), (4, 4, 8))
        self.tensorized_shape = tltorch.utils.get_tensorized_shape(in_features, out_features, verbose=False)

class CP(LowRankTensor):

    def __init__(self, in_features, out_features, max_rank, prior_type='log_uniform', eta=None, device=None, dtype=None):
        '''
        arguments:
            in_features: an int, the input size
            out_features: an int, the output size
            max_rank: a float, if representing compression ratio; 
                      an int, if representing actual tensor rank
            tensor_type: a str, the layer's tensor decomposition:
                         'CP', 'Tucker', 'TT', or 'TTM'
            prior_type: a str, the prior distribution of the rank parameter:
                        'log_uniform' or 'half_cauchy'
            eta: a float, the hyper-parameter if prior_type == 'half_cauchy'
        '''

        
        super().__init__(in_features, out_features, prior_type, eta, device, dtype)

        # get tltorch CP tensor
        # it will have len(tensorized_in) + len(tensorized_out) factor matrices
        # with the shapes of (tensorized_in[i], rank) and (tensorized_out[i], rank)
        self.tensor = tltorch.TensorizedTensor.new(tensorized_shape=self.tensorized_shape,
                                                   rank=max_rank,
                                                   factorization='CP',
                                                   device=device,
                                                   dtype=dtype)

        # set self.max_rank to tensor.rank because max_rank might be a float
        self.max_rank = self.tensor.rank
        # set target variance for factor initialization
        target_var = 1 / in_features
        factor_std = (target_var / self.tensor.rank) ** (1 / (4 * self.tensor.order))
        # initialize the factors
        for factor in self.tensor.factors:
            nn.init.normal_(factor, 0, factor_std)
        # initialize the rank parameter with the shape of (self.max_rank,)
        # for each columns of the factor matrices
        self.rank_parameter = nn.Parameter(torch.rand((self.max_rank,), device=device, dtype=dtype))

    def get_log_prior(self):

        with torch.no_grad():
            self.rank_parameter[:] = self.rank_parameter.clamp(1e-30, 1e30)
        
        # self.threshold(self.rank_parameter)
        log_prior = torch.sum(self.rank_parameter_prior_distribution.log_prob(self.rank_parameter))
        
        # 0 mean normal distribution for the factors
        factor_prior_distribution = Normal(0, self.rank_parameter)
        for factor in self.tensor.factors:
            log_prior = log_prior + factor_prior_distribution.log_prob(factor).sum(0).sum()
        
        return log_prior
    
    def estimate_rank(self):
        
        rank = 0
        for factor in self.tensor.factors:
            factor_rank = torch.sum(factor.var(axis=0) > 1e-5)
            rank = max(rank, factor_rank)
        
        return rank

class TT(LowRankTensor):

    def __init__(self, in_features, out_features, max_rank, prior_type='log_uniform', eta=None, device=None, dtype=None):
        '''
        max_rank: an 'int' or a 'float' if indicating comprssion ratio
        '''

        super().__init__(in_features, out_features, prior_type, eta, device, dtype)

        self.tensorized_dim = [*self.tensorized_shape[0], *self.tensorized_shape[1]]
            
        self.tensor = tltorch.TTTensor.new(shape=self.tensorized_dim, rank=max_rank, device=device, dtype=dtype)
        self.max_rank = self.tensor.rank
        target_var = 1 / in_features
        factor_std = ((target_var / np.prod(self.tensor.rank)) ** (1 / self.tensor.order)) ** 0.5

        for factor in self.tensor.factors:
            nn.init.normal_(factor, 0, factor_std)

        self.rank_parameter = nn.ParameterList([nn.Parameter(torch.rand((self.max_rank[1+i],), device=device, dtype=dtype)) \
            for i in range(len(self.tensorized_dim)-1)])

    def get_log_prior(self):

        log_prior = 0.0
        for r in self.rank_parameter:
            # clamp rank_param because <=0 is undefined 
            with torch.no_grad():
                r[:] = r.clamp(1e-10, 1e10)
            log_prior = log_prior + torch.sum(self.rank_parameter_prior_distribution.log_prob(r))

        
        # 0 mean normal distribution for the factors
        for i in range(len(self.rank_parameter)):
            factor_prior_distribution = Normal(0, self.rank_parameter[i])
            log_prior = log_prior + factor_prior_distribution.log_prob(self.tensor.factors[i]).sum((0,1)).sum()
        
        factor_prior_distribution = Normal(0, self.rank_parameter[-1])
        log_prior = log_prior + factor_prior_distribution.log_prob(self.tensor.factors[-1]).sum((1,2)).sum()

        return log_prior
    
    def estimate_rank(self):

        rank = [1]
        for factor in self.tensor.factors[:-1]:
            rank.append(torch.sum(factor.var((0,1)) > 1e-4).item())
        
        #rank.append(torch.sum(self.tensor.factors[-1].var((1,2) > 1e-5)))
        rank.append(1)

        return rank