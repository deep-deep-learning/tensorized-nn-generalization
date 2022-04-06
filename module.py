import torch.nn as nn

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

        