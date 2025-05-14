import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.AdaMSHyper_layers import (AvgPooling_Construct,
                                      Bottleneck_Construct, Conv_Construct,
                                      Decoder, EncoderLayer,
                                      MaxPooling_Construct, Predictor)
from layers.Embed import CustomEmbedding, DataEmbedding
from torch_geometric.data import data as D
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import degree, softmax
from utils.tools import PParameter


class Model(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.channels = configs.enc_in

        self.all_size = get_mask(configs.seq_len, configs.window_size)
        self.Ms_length = sum(self.all_size)

        self.multiadphyper = multi_adaptive_hypergraoh(configs)
        self.conv_layers = eval(configs.CSCM)(configs.enc_in, configs.window_size, configs.enc_in)

        self.hyper_num = configs.hyper_num
        self.hyconv = nn.ModuleList()
        for i in range (len(self.hyper_num)):
            self.hyconv.append(HypergraphConv(configs.enc_in, configs.enc_in))
        self.hyperedge_atten = SelfAttentionLayer(configs)

        self.individual = configs.individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Tran = nn.Linear(self.pred_len, self.pred_len)

        self.out_tran = nn.Linear(self.Ms_length, self.pred_len)
        self.out_tran.weight = nn.Parameter((1 / self.Ms_length) * torch.ones([self.pred_len, self.Ms_length]))
        self.inter_tran = nn.Linear(80, self.pred_len)

    def forward(self, x, x_mark_enc, *args, **kwargs):
        # normalization
        mean_enc = x.mean(1, keepdim=True).detach()
        x = x - mean_enc
        std_enc = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = x / std_enc

        # calculate hypergraph of each scale
        adj_matrix = self.multiadphyper(x)
        # calculate feature of each scale, MFE module
        seq_enc = self.conv_layers(x)

        sum_hyper_list = []
        for i in range(len(self.hyper_num)):
            mask = torch.tensor(adj_matrix[i]).to(x.device)  # [2, S], S <= N * M

            # inter-scale
            node_value = seq_enc[i].permute(0, 2, 1)  # [B, Dx, scale]
            node_value = torch.tensor(node_value).to(x.device)
            edge_sums = {}
            for edge_id, node_id in zip(mask[1], mask[0]):
                edge_id = edge_id.item()
                node_id = node_id.item()
                if edge_id not in edge_sums:
                    edge_sums[edge_id] = node_value[:, :, node_id]  # [B, Dx]
                else:
                    edge_sums[edge_id] += node_value[:, :, node_id]

            for edge_id, sum_value in edge_sums.items():
                sum_value = sum_value.unsqueeze(1)  # [B, 1, Dx]
                sum_hyper_list.append(sum_value)

            # intra-scale
            output, constrainloss = self.hyconv[i](seq_enc[i], mask)

            if i == 0:
                result_tensor = output
                result_conloss = constrainloss
            else:
                result_tensor = torch.cat((result_tensor, output), dim=1)
                result_conloss += constrainloss

        sum_hyper_list = torch.cat(sum_hyper_list, dim=1)
        sum_hyper_list = sum_hyper_list.to(x.device)

        padding_need = 80 - sum_hyper_list.size(1)
        hyperedge_attention = self.hyperedge_atten(sum_hyper_list)
        pad = F.pad(hyperedge_attention, (0, 0, 0, padding_need, 0, 0))

        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0,2,1))
            x_out = self.out_tran(result_tensor.permute(0, 2, 1))  # ori
            x_out_inter = self.inter_tran(pad.permute(0, 2, 1))

        x = x_out + x + x_out_inter
        x = self.Linear_Tran(x).permute(0, 2, 1)
        x = x * std_enc + mean_enc

        return x, abs(result_conloss)  # [Batch, Output length, Channel]


class HypergraphConv(MessagePassing):
    def __init__(
        self, in_channels, out_channels, use_attention=True, heads=1, concat=True, negative_slope=0.2,
        dropout=0.1, bias=False
    ):
        super(HypergraphConv, self).__init__(aggr='add')
        self.soft = nn.Softmax(dim=0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention

        self.gamma = 4.2

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout

            self.weight = PParameter(torch.Tensor(in_channels, out_channels))
            self.att = PParameter(torch.Tensor(1, heads, 2 * int(out_channels / heads)))  # [1, 1, 2*Dx]
        else:
            self.heads = 1
            self.concat = True
            self.weight = PParameter(torch.Tensor(in_channels, out_channels))

        if bias and concat:
            self.bias = PParameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = PParameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)

    def __forward__(self, x, hyperedge_index, alpha=None):
        D = degree(hyperedge_index[0], x.size(0), x.dtype)
        num_edges = 2 * (hyperedge_index[1].max().item() + 1)
        B = 1.0 / degree(hyperedge_index[1], int(num_edges/2), x.dtype)
        B[B == float("inf")] = 0

        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha)
        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha)

        return out

    def message(self, x_j, edge_index_i, norm, alpha):
        out = norm[edge_index_i].view(-1, 1, 1) * x_j
        if alpha is not None:
            out = alpha.unsqueeze(-1) * out
        return out

    def forward(self, x, hyperedge_index):
        # x: [B, scale, Dx], hyperedge_index: [2, S]
        x = torch.matmul(x, self.weight)  # [B, scale, Dx]
        x1 = x.transpose(0, 1)  # [scale, B, Dx]
        # node representation
        x_i = torch.index_select(x1, dim=0, index=hyperedge_index[0])  # [S, B, Dx]

        edge_sums = {}
        for edge_id, node_id in zip(hyperedge_index[1], hyperedge_index[0]):
            edge_id = edge_id.item()
            node_id = node_id.item()
            if edge_id not in edge_sums:
                edge_sums[edge_id] = x1[node_id, :, :]  # [B, Dx]
            else:
                edge_sums[edge_id] += x1[node_id, :, :]

        result_list = torch.stack([value for value in edge_sums.values()], dim=0)  # [M', B, Dx]
        # hyperedge representation
        x_j = torch.index_select(result_list, dim=0, index=hyperedge_index[1])  # [S, B, Dx]

        loss_hyper = 0
        for k in range(len(edge_sums)):
            for m in range(len(edge_sums)):
                inner_product = torch.sum(edge_sums[k] * edge_sums[m], dim=1, keepdim=True)  # [B, 1]
                norm_q_i = torch.norm(edge_sums[k], dim=1, keepdim=True)  # [B, 1]
                norm_q_j = torch.norm(edge_sums[m], dim=1, keepdim=True)  # [B, 1]
                alpha = inner_product / (norm_q_i * norm_q_j)
                distan = torch.norm(edge_sums[k] - edge_sums[m], dim=1, keepdim=True)  # [B, 1]
                loss_item = alpha * distan + (1 - alpha) * (torch.clamp(torch.tensor(self.gamma) - distan, min=0.0))
                loss_hyper += torch.abs(torch.mean(loss_item))

        loss_hyper = loss_hyper / ((len(edge_sums) + 1) ** 2)

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)  # [S, B]
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, hyperedge_index[0], num_nodes=x1.size(0))
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        D = degree(hyperedge_index[0], x1.size(0), x.dtype)  # [N] or [scale]
        # num_edges = 2 * (hyperedge_index[1].max().item() + 1)
        # B = 1.0 / degree(hyperedge_index[1], int(num_edges / 2),x.dtype)
        num_edges = hyperedge_index[1].max().item() + 1
        B = 1.0 / degree(hyperedge_index[1], num_edges, x.dtype)
        B[B == float("inf")] = 0

        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x1, norm=B, alpha=alpha)
        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha)

        out = out.transpose(0, 1)
        constrain_loss = x_i - x_j
        constrain_lossfin1 = torch.mean(constrain_loss)
        constrain_losstotal = abs(constrain_lossfin1) + loss_hyper

        return out, constrain_losstotal

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels, self.out_channels)


class multi_adaptive_hypergraoh(nn.Module):
    def __init__(self, configs):
        super(multi_adaptive_hypergraoh, self).__init__()
        self.seq_len = configs.seq_len

        self.window_size = configs.window_size
        self.inner_size = configs.inner_size

        self.dim = configs.d_model
        self.hyper_num = configs.hyper_num

        self.alpha = 3
        self.k = configs.k
        self.beta = 0.5

        self.node_num = get_mask(self.seq_len, self.window_size)

        self.embedhy = nn.ModuleList()
        self.embednod = nn.ModuleList()
        for i in range(len(self.hyper_num)):
            self.embedhy.append(nn.Embedding(self.hyper_num[i], self.dim))
            self.embednod.append(nn.Embedding(self.node_num[i], self.dim))

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        hyperedge_all = []
        for i in range(len(self.hyper_num)):
            M = self.hyper_num[i]  # number of hyperedges
            N = self.node_num[i]  # number of nodes

            hypidxc = torch.arange(M).to(x.device)
            nodeidx = torch.arange(N).to(x.device)
            hyperen = self.embedhy[i](hypidxc)  # [M, d_model]
            nodeec = self.embednod[i](nodeidx)  # [N, d_model]

            a = torch.mm(nodeec, hyperen.transpose(1, 0))  # [N, M]
            adj = F.softmax(F.relu(self.alpha * a))

            # Top-K sparsification
            mask = torch.zeros(N, M).to(x.device)  # [N, M]
            s1, t1 = adj.topk(min(M, self.k), 1)
            mask.scatter_(1, t1, s1.fill_(1))
            adj = adj * mask

            # beta sparsification
            adj = torch.where(adj > self.beta, torch.tensor(1).to(x.device), torch.tensor(0).to(x.device))
            adj = adj[:, (adj != 0).any(dim=0)]  # [N, M'], remove empty hyperedges

            matrix_array = torch.tensor(adj, dtype=torch.int)
            # find the node index of each hyperedge
            result_list = [
                list(torch.nonzero(matrix_array[:, col]).flatten().tolist())
                for col in range(matrix_array.shape[1])
            ]
            # node index
            node_list = torch.cat([torch.tensor(sublist) for sublist in result_list if len(sublist) > 0]).tolist()
            # count the number of nodes in each hyperedge
            count_list = list(torch.sum(adj, dim=0).tolist())
            # hyperedge index
            hperedge_list = torch.cat([torch.full((count,), idx) for idx, count in enumerate(count_list, start=0)]).tolist()

            hypergraph = np.vstack((node_list, hperedge_list))  # [2, S], S <= N * M
            hyperedge_all.append(hypergraph)

        return hyperedge_all


class SelfAttentionLayer(nn.Module):
    def __init__(self, configs):
        super(SelfAttentionLayer, self).__init__()
        self.query_weight = nn.Linear(configs.enc_in, configs.enc_in)
        self.key_weight = nn.Linear(configs.enc_in, configs.enc_in)
        self.value_weight = nn.Linear(configs.enc_in, configs.enc_in)

    def forward(self, x):
        q = self.query_weight(x)
        k = self.key_weight(x)
        v = self.value_weight(x)
        attention_scores = F.softmax(torch.matmul(q, k.transpose(1, 2)) / (k.shape[-1] ** 0.5), dim=-1)
        attended_values = torch.matmul(attention_scores, v)

        return attended_values


def get_mask(input_size, window_size):
    """Get the attention mask of HyperGraphConv"""
    # Get the size of all layers
    all_size = []
    all_size.append(input_size)
    for i in range(len(window_size)):
        layer_size = math.floor(all_size[i] / window_size[i])
        all_size.append(int(layer_size))
    return all_size
