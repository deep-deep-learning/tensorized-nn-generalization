import torch.nn.functional as F
import torch.nn as nn
from module import AdaptiveRankLinear

class AutoNet(nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size, rank, device=None, dtype=None):

        super().__init__()

        self.layer_1 = AdaptiveRankLinear(input_size, hidden_sizes[0], rank, device=device, dtype=dtype)
        self.layer_2 = AdaptiveRankLinear(hidden_sizes[0], hidden_sizes[1], rank, device=device, dtype=dtype)
        self.layer_3 = nn.Linear(hidden_sizes[1], output_size, device=device, dtype=dtype)
    
    def forward(self, x):

        output = F.relu(self.layer_1(x))
        output = F.relu(self.layer_2(output))
        output = self.layer_3(output)

        return output