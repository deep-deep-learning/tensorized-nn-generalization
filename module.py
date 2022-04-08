import torch
import torch.nn as nn
import low_rank_tensor as LowRankTensor
from tensor_times_matrix import tensor_times_matrix_fwd

class AdaptiveRankLinear(nn.Module):
    
    def __init__(self, in_features, out_features, max_rank, bias=True,
                 tensor_type='TT', prior_type='log_uniform', eta=None,
                 device=None, dtype=None):
        '''
        arguments:
            in_features: an int, the input size
            out_features: an int, the output size
            max_rank: a float, if representing compression ratio; 
                      an int, if representing actual tensor rank
            bias: True, if the layer has a bias; False, otherwise
            tensor_type: a str, the layer's tensor decomposition:
                         'CP', 'Tucker', 'TT', or 'TTM'
            prior_type: a str, the prior distribution of the rank parameter:
                        'log_uniform' or 'half_cauchy'
            eta: a float, the hyper-parameter if prior_type == 'half_cauchy'
        '''

        super().__init__()

        self.weight_tensor = getattr(LowRankTensor, tensor_type)(in_features, out_features, max_rank, prior_type, eta, device, dtype)
        
       # initialize bias
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_features,), device=device, dtype=dtype))
        else:
            self.bias = None

    def forward(self, x):

        output = tensor_times_matrix_fwd(self.weight_tensor.tensor, x.T)

        if self.bias is not None:
            output = output + self.bias
            
        return output

    def get_log_prior(self):

        return self.weight_tensor.get_log_prior()

    def estimate_rank(self):

        return self.weight_tensor.estimate_rank()