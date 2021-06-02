import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import (
    Sequential,
    Linear,
    ReLU,
    GRU,
    Embedding,
    BatchNorm1d,
    Dropout,
    LayerNorm,
)
from torch_geometric.nn import (
    Set2Set,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
)


# Sine matrix with neural network
class SM(torch.nn.Module):
    def __init__(self, data, dim1=64, fc_count=1,  **kwargs):
        super(SM, self).__init__()
        
        self.lin1 = torch.nn.Linear(data[0].extra_features_SM.shape[1], dim1)

        self.lin_list = torch.nn.ModuleList(
            [torch.nn.Linear(dim1, dim1) for i in range(fc_count)]
        )

        self.lin2 = torch.nn.Linear(dim1, 1)

    def forward(self, data):

        out = F.relu(self.lin1(data.extra_features_SM))
        for layer in self.lin_list:
            out = F.relu(layer(out))
        out = self.lin2(out)
        if out.shape[1] == 1:
            return out.view(-1)
        else:
            return out


# Smooth Overlap of Atomic Positions with neural network
class SOAP(torch.nn.Module):
    def __init__(self, data, dim1, fc_count,  **kwargs):
        super(SOAP, self).__init__()
        
        self.lin1 = torch.nn.Linear(data[0].extra_features_SOAP.shape[1], dim1)

        self.lin_list = torch.nn.ModuleList(
            [torch.nn.Linear(dim1, dim1) for i in range(fc_count)]
        )

        self.lin2 = torch.nn.Linear(dim1, 1)

    def forward(self, data):

        out = F.relu(self.lin1(data.extra_features_SOAP))
        for layer in self.lin_list:
            out = F.relu(layer(out))
        out = self.lin2(out)
        if out.shape[1] == 1:
            return out.view(-1)
        else:
            return out
