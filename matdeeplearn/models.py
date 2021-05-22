import torch
import torch_geometric
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Sequential, Linear, ReLU, GRU, Embedding, BatchNorm1d
from torch_geometric.nn import (
    NNConv,
    Set2Set,
    CGConv,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
    SchNet,
    BatchNorm,
    GraphConv,
    MessagePassing,
    MetaLayer,
    GCNConv,
)
from torch_geometric.data import DataLoader, Dataset, Data
from torch_geometric.utils import remove_self_loops, dense_to_sparse, degree
from torch_scatter import scatter_mean, scatter_add, scatter_max, scatter
from torch_geometric.nn.models.schnet import InteractionBlock

################################################################################
# ML Models
################################################################################

# Models included here:
# CGCNN, MPNN, SchNet, MEGNet, GCN, Standard NN using SM and SOAP

# Simple GCN
class GCN_net(torch.nn.Module):
    def __init__(self, data, dim1, dim2, conv_count, fc_count, pool, **kwargs):
        super(GCN_net, self).__init__()

        self.pool = pool
        self.lin0 = torch.nn.Linear(data.num_features, dim1)
        self.conv_list = torch.nn.ModuleList(
            [
                GCNConv(dim1, dim1, improved=True, add_self_loops=False)
                for i in range(conv_count)
            ]
        )

        if self.pool == "set2set":
            self.set2set = Set2Set(dim1, processing_steps=3)
            self.lin1 = torch.nn.Linear(dim1 * 2, dim2)
        else:
            self.lin1 = torch.nn.Linear(dim1, dim2)

        self.lin_list = torch.nn.ModuleList(
            [torch.nn.Linear(dim2, dim2) for i in range(fc_count)]
        )
        self.lin2 = torch.nn.Linear(dim2, 1)

    def forward(self, data):

        out = F.relu(self.lin0(data.x))
        for layer in self.conv_list:
            out = F.relu(layer(out, data.edge_index, data.edge_weight))

        if self.pool == "set2set":
            out = self.set2set(out, data.batch)
        else:
            out = getattr(torch_geometric.nn, self.pool)(out, data.batch)

        out = F.relu(self.lin1(out))
        for layer in self.lin_list:
            out = F.relu(layer(out))
        out = self.lin2(out)
        return out.view(-1)


# MPNN
class MPNN(torch.nn.Module):
    def __init__(self, data, dim1, dim2, dim3, conv_count, fc_count, pool, **kwargs):
        super(MPNN, self).__init__()

        self.pool = pool
        self.lin0 = torch.nn.Linear(data.num_features, dim1)
        self.conv_list = torch.nn.ModuleList()
        self.gru_list = torch.nn.ModuleList()
        for i in range(conv_count):
            nn = Sequential(
                Linear(data.num_edge_features, dim3), ReLU(), Linear(dim3, dim1 * dim1)
            )
            conv = NNConv(dim1, dim1, nn, aggr="mean")
            gru = GRU(dim1, dim1)
            self.conv_list.append(conv)
            gru = GRU(dim1, dim1)
            self.gru_list.append(gru)

        if self.pool == "set2set":
            self.set2set = Set2Set(dim1, processing_steps=3)
            self.lin1 = torch.nn.Linear(dim1 * 2, dim2)
        else:
            self.lin1 = torch.nn.Linear(dim1, dim2)

        self.lin_list = torch.nn.ModuleList(
            [torch.nn.Linear(dim2, dim2) for i in range(fc_count)]
        )
        self.lin2 = torch.nn.Linear(dim2, 1)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)
        for i in range(len(self.conv_list)):
            m = F.relu(self.conv_list[i](out, data.edge_index, data.edge_attr))
            out, h = self.gru_list[i](m.unsqueeze(0), h)
            out = out.squeeze(0)

        if self.pool == "set2set":
            out = self.set2set(out, data.batch)
        else:
            out = getattr(torch_geometric.nn, self.pool)(out, data.batch)

        out = F.relu(self.lin1(out))
        for layer in self.lin_list:
            out = F.relu(layer(out))
        out = self.lin2(out)
        return out.view(-1)


# Schnet
class SchNet(SchNet):
    def __init__(
        self, data, dim1, dim2, dim3, conv_count, fc_count, pool, cutoff=8, **kwargs
    ):
        super(SchNet, self).__init__()

        self.pool = pool
        self.lin0 = torch.nn.Linear(data.num_features, dim1)

        self.interactions = torch.nn.ModuleList()
        for _ in range(conv_count):
            block = InteractionBlock(dim1, data.num_edge_features, dim2, cutoff)
            self.interactions.append(block)

        if self.pool == "set2set":
            self.set2set = Set2Set(dim1, processing_steps=3)
            self.lin1 = torch.nn.Linear(dim1 * 2, dim3)
        else:
            self.lin1 = torch.nn.Linear(dim1, dim3)

        self.lin_list = torch.nn.ModuleList(
            [torch.nn.Linear(dim3, dim3) for i in range(fc_count)]
        )
        self.lin2 = torch.nn.Linear(dim3, 1)

    def forward(self, data):

        out = F.relu(self.lin0(data.x))

        for interaction in self.interactions:
            out = out + interaction(
                out, data.edge_index, data.edge_weight, data.edge_attr
            )

        if self.pool == "set2set":
            out = self.set2set(out, data.batch)
        else:
            out = getattr(torch_geometric.nn, self.pool)(out, data.batch)

        out = F.relu(self.lin1(out))
        for layer in self.lin_list:
            out = F.relu(layer(out))
        out = self.lin2(out)

        return out.view(-1)


# CGCNN
class CGCNN(torch.nn.Module):
    def __init__(self, data, dim1, dim2, conv_count, fc_count, pool, **kwargs):
        super(CGCNN, self).__init__()

        self.pool = pool
        self.lin0 = torch.nn.Linear(data.num_features, dim1)
        self.conv_list = torch.nn.ModuleList(
            [
                CGConv(dim1, data.num_edge_features, aggr="mean", batch_norm=True)
                for i in range(conv_count)
            ]
        )

        if self.pool == "set2set":
            self.set2set = Set2Set(dim1, processing_steps=3)
            self.lin1 = torch.nn.Linear(dim1 * 2, dim2)
        else:
            self.lin1 = torch.nn.Linear(dim1, dim2)

        self.lin_list = torch.nn.ModuleList(
            [torch.nn.Linear(dim2, dim2) for i in range(fc_count)]
        )
        self.lin2 = torch.nn.Linear(dim2, 1)

    def forward(self, data):

        out = F.relu(self.lin0(data.x))
        for layer in self.conv_list:
            out = F.relu(layer(out, data.edge_index, data.edge_attr))

        if self.pool == "set2set":
            out = self.set2set(out, data.batch)
        else:
            out = getattr(torch_geometric.nn, self.pool)(out, data.batch)

        out = F.relu(self.lin1(out))
        for layer in self.lin_list:
            out = F.relu(layer(out))
        out = self.lin2(out)
        return out.view(-1)


# Megnet
class Megnet_EdgeModel(torch.nn.Module):
    def __init__(self, dim):
        super(Megnet_EdgeModel, self).__init__()
        self.edge_mlp_1 = Sequential(Linear(dim * 4, dim), ReLU(), Linear(dim, dim))

    def forward(self, src, dest, edge_attr, u, batch):
        comb = torch.cat([src, dest, edge_attr, u[batch]], dim=1)
        out = self.edge_mlp_1(comb)
        return out


class Megnet_NodeModel(torch.nn.Module):
    def __init__(self, dim):
        super(Megnet_NodeModel, self).__init__()
        self.node_mlp_1 = Sequential(Linear(dim * 3, dim), ReLU(), Linear(dim, dim))

    def forward(self, x, edge_index, edge_attr, u, batch):
        # row, col = edge_index
        v_e = scatter_mean(edge_attr, edge_index[0, :], dim=0)
        comb = torch.cat([x, v_e, u[batch]], dim=1)
        out = self.node_mlp_1(comb)
        return out


class Megnet_GlobalModel(torch.nn.Module):
    def __init__(self, dim):
        super(Megnet_GlobalModel, self).__init__()
        self.global_mlp_1 = Sequential(Linear(dim * 3, dim), ReLU(), Linear(dim, dim))

    def forward(self, x, edge_index, edge_attr, u, batch):
        u_e = scatter_mean(edge_attr, edge_index[0, :], dim=0)
        u_e = scatter_mean(u_e, batch, dim=0)
        u_v = scatter_mean(x, batch, dim=0)
        comb = torch.cat([u_e, u_v, u], dim=1)
        out = self.global_mlp_1(comb)
        return out


class MEGNet(torch.nn.Module):
    def __init__(self, data, dim1, dim2, dim3, conv_count, fc_count, pool, **kwargs):
        super(MEGNet, self).__init__()
        self.lin0 = torch.nn.Linear(data.num_node_features, dim1)
        self.pool = pool
        megnet_block = MetaLayer(
            Megnet_EdgeModel(dim2), Megnet_NodeModel(dim2), Megnet_GlobalModel(dim2)
        )
        self.e_embed_list = torch.nn.ModuleList()
        self.x_embed_list = torch.nn.ModuleList()
        self.u_embed_list = torch.nn.ModuleList()
        self.meg_list = torch.nn.ModuleList()

        for i in range(0, conv_count):
            if i == 0:
                meg = megnet_block
                e_embed = Sequential(
                    Linear(data.num_edge_features, dim1), ReLU(), Linear(dim1, dim2)
                )
                x_embed = Sequential(Linear(dim1, dim1), ReLU(), Linear(dim1, dim2))
                u_embed = Sequential(
                    Linear((data[0].u.shape[1]), dim1), ReLU(), Linear(dim1, dim2)
                )
                self.e_embed_list.append(e_embed)
                self.x_embed_list.append(x_embed)
                self.u_embed_list.append(u_embed)
                self.meg_list.append(meg)
            elif i > 0:
                meg = megnet_block
                e_embed = Sequential(Linear(dim2, dim1), ReLU(), Linear(dim1, dim2))
                x_embed = Sequential(Linear(dim2, dim1), ReLU(), Linear(dim1, dim2))
                u_embed = Sequential(Linear(dim2, dim1), ReLU(), Linear(dim1, dim2))
                self.e_embed_list.append(e_embed)
                self.x_embed_list.append(x_embed)
                self.u_embed_list.append(u_embed)
                self.meg_list.append(meg)

        if self.pool == "set2set":
            self.set2set_x = Set2Set(dim2, processing_steps=3)
            self.set2set_e = Set2Set(dim2, processing_steps=3)
            self.lin1 = torch.nn.Linear(dim2 * 5, dim3)

        else:
            self.lin1 = torch.nn.Linear(dim2 * 3, dim3)

        self.lin_list = torch.nn.ModuleList(
            [torch.nn.Linear(dim3, dim3) for i in range(fc_count)]
        )
        self.lin2 = torch.nn.Linear(dim3, 1)

    def forward(self, data):

        x = F.relu(self.lin0(data.x))

        for i in range(0, len(self.meg_list)):

            if i == 0:
                e_temp = self.e_embed_list[i](data.edge_attr)
                x_temp = self.x_embed_list[i](x)
                u_temp = self.u_embed_list[i](data.u)
                x_out, e_out, u_out = self.meg_list[i](
                    x_temp, data.edge_index, e_temp, u_temp, data.batch
                )
                x = torch.add(x_out, x_temp)
                e = torch.add(e_out, e_temp)
                u = torch.add(u_out, u_temp)

            elif i > 0:
                e_temp = self.e_embed_list[i](e)
                x_temp = self.x_embed_list[i](x)
                u_temp = self.u_embed_list[i](u)
                x_out, e_out, u_out = self.meg_list[i](
                    x_temp, data.edge_index, e_temp, u_temp, data.batch
                )
                x = torch.add(x_out, x)
                e = torch.add(e_out, e)
                u = torch.add(u_out, u)

        if self.pool == "set2set":
            x_pool = self.set2set_x(x, data.batch)
            # not exactly same as original, extra scatter operation to go from edge to node index
            e = scatter(e, data.edge_index[0, :], dim=0, reduce="mean")
            e_pool = self.set2set_e(e, data.batch)
            comb_pool = torch.cat([x_pool, e_pool, u], dim=1)

        else:
            x_pool = scatter(x, data.batch, dim=0, reduce=self.pool)
            e_pool = scatter(e, data.edge_index[0, :], dim=0, reduce=self.pool)
            e_pool = scatter(e_pool, data.batch, dim=0, reduce=self.pool)
            comb_pool = torch.cat([x_pool, e_pool, u], dim=1)

        out = F.relu(self.lin1(comb_pool))
        for layer in self.lin_list:
            out = F.relu(layer(out))
        out = self.lin2(out)

        return out.view(-1)


# Sine matrix with neural network
class SM(torch.nn.Module):
    def __init__(self, data, dim1, fc_count, **kwargs):
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
        return out.view(-1)


# Smooth Overlap of Atomic Positions with neural network
class SOAP(torch.nn.Module):
    def __init__(self, data, dim1, fc_count, **kwargs):
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
        return out.view(-1)


# Prints model summary
def model_summary(model):
    model_params_list = list(model.named_parameters())
    print("--------------------------------------------------------------------------")
    line_new = "{:>30}  {:>20} {:>20}".format(
        "Layer.Parameter", "Param Tensor Shape", "Param #"
    )
    print(line_new)
    print("--------------------------------------------------------------------------")
    for elem in model_params_list:
        p_name = elem[0]
        p_shape = list(elem[1].size())
        p_count = torch.tensor(elem[1].size()).prod().item()
        line_new = "{:>30}  {:>20} {:>20}".format(p_name, str(p_shape), str(p_count))
        print(line_new)
    print("--------------------------------------------------------------------------")
    total_params = sum([param.nelement() for param in model.parameters()])
    print("Total params:", total_params)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params:", num_trainable_params)
    print("Non-trainable params:", total_params - num_trainable_params)
