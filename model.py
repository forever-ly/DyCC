import math
import torch
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
import torch.nn as nn
import copy
import numpy as np
from graph_trans_model import TransformerNodeEncoder_v3, TransformerNodeDecoder
from torch_geometric.utils import to_dense_batch
from pos_enc.encoder import PosEncoder
from torch_scatter import scatter_mean, scatter_add, scatter_std
def get_prototype(num_proto,output_dim,device):
    prototypes = torch.empty(num_proto, output_dim)
    _sqrt_k = (1. / output_dim) ** 0.5
    torch.nn.init.uniform_(prototypes, -_sqrt_k, _sqrt_k)
    prototypes = torch.nn.parameter.Parameter(prototypes)
    # -- init prototype labels
    proto_labels = one_hot(torch.tensor([i for i in range(num_proto)]), num_proto,device)
    return prototypes,proto_labels
import torch_geometric
import random

def init_msn_loss(
    num_views=1,
    tau=0.1,
    me_max=True,
    return_preds=False
):
    """
    Make unsupervised MSN loss

    :num_views: number of anchor views
    :param tau: cosine similarity temperature
    :param me_max: whether to perform me-max regularization
    :param return_preds: whether to return anchor predictions
    """
    softmax = torch.nn.Softmax(dim=1)

    def sharpen(p, T):
        sharp_p = p**(1./T)
        sharp_p /= torch.sum(sharp_p, dim=1, keepdim=True)
        return sharp_p

    def snn(query, supports, support_labels, temp=tau):
        """ Soft Nearest Neighbours similarity classifier """
        query = torch.nn.functional.normalize(query)
        supports = torch.nn.functional.normalize(supports)
        return softmax(query @ supports.T / temp) @ support_labels

    def loss(
        anchor_views,
        target_views,
        prototypes,
        proto_labels,
        T=0.25,
        use_entropy=False,
        use_sinkhorn=False,
        sharpen=sharpen,
        snn=snn
    ):
        # Step 1: compute anchor predictions
        probs = snn(anchor_views, prototypes, proto_labels)
        # Step 2: compute targets for anchor predictions
        with torch.no_grad():
            targets = sharpen(snn(target_views, prototypes, proto_labels), T=T)
            targets = torch.cat([targets for _ in range(num_views)], dim=0)

        print("==" * 10, torch.argmax(targets,dim=1))
        print("==" * 10, torch.argmax(targets,dim=1).shape)
        print("==" * 10, torch.mean(targets, dim=0))
        print("==" * 10, torch.mean(targets, dim=0).shape)
        with open('/home/zjh/remote/KDD/SimSGT-main/chem/target2.txt', 'a') as file:
            hard_label = torch.argmax(targets,dim=1)
            for i in hard_label:
                file.write(str(i.detach().cpu().item())+'\n')


        with open('/home/zjh/remote/KDD/SimSGT-main/chem/target_soft2.txt', 'a') as file:
            soft_label = torch.mean(targets, dim=0)

            numpy_data = soft_label.detach().cpu().numpy()
            csv_data = np.array2string(numpy_data, separator=',', formatter={'float_kind': lambda x: "%.6f" % x})

            file.write(csv_data+"\n")
        # per_target = perplexity(targets)
        # per_probs = perplexity(probs)
        # print("==" * 10, per_target)
        # print("==" * 10, per_probs)
        # with open('/home/zjh/remote/KDD/SimSGT-main/chem/pre2.txt', 'a') as file:
        #
        #     file.write(str(per_target.detach().cpu().item())+","+str(per_target.detach().cpu().item())+ '\n')

        # Step 3: compute cross-entropy loss H(targets, queries)
        loss = torch.mean(torch.sum(torch.log(probs**(-targets)), dim=1))

        # Step 4: compute me-max regularizer
        rloss = 0.
        if me_max:
            avg_probs = torch.mean(probs, dim=0)
            rloss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))

        sloss = 0.
        if use_entropy:
            sloss = torch.mean(torch.sum(torch.log(probs**(-probs)), dim=1))

        # -- logging
        with torch.no_grad():
            num_ps = float(len(set(targets.argmax(dim=1).tolist())))
            max_t = targets.max(dim=1).values.mean()
            min_t = targets.min(dim=1).values.mean()
            log_dct = {'np': num_ps, 'max_t': max_t, 'min_t': min_t}

        if return_preds:
            return loss, rloss, sloss, log_dct, targets

        return loss, rloss, sloss, log_dct

    return loss


def perplexity(probs):
    # 计算每行的概率分布的对数
    avg_probs = torch.mean(probs, dim=0)
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
    return perplexity
def one_hot(targets, num_classes, device,smoothing=0.0):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    targets = targets.long().view(-1, 1).to(device)
    return torch.full((len(targets), num_classes), off_value, device=device).scatter_(1, targets, on_value)

num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3

class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 
        

    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, out_dim, aggr = "add", act_func='relu', **kwargs):
        kwargs.setdefault('aggr', aggr)
        self.aggr = aggr
        super(GINConv, self).__init__(**kwargs)
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), get_activation(act_func), torch.nn.Linear(2*emb_dim, out_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.add_selfloop = False

    def forward(self, x, edge_index, edge_attr):
        # #add self loops in the edge space
        if self.add_selfloop:
            edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))
            #add features corresponding to self-loop edges.
            self_loop_attr = torch.zeros(x.size(0), 2)
            self_loop_attr[:,0] = 4 #bond type for self-loop edge
            self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
            edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        # return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class NonParaGINConv(MessagePassing):
    ## non-parametric gin
    def __init__(self, eps, aggr="add", **kwargs):
        kwargs.setdefault('aggr', aggr)
        super().__init__(**kwargs)
        self.aggr = aggr
        self.eps = eps

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x) + x * self.eps

    def message(self, x_j):
        return x_j


class NonParaGCNConv(MessagePassing):
    ## non-parametric gcn
    def __init__(self, aggr="add", **kwargs):
        kwargs.setdefault('aggr', aggr)
        super().__init__(**kwargs)
        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index):
        norm = self.norm(edge_index, x.size(0), x.dtype)
        return self.propagate(edge_index, x=x, norm=norm) + x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


# 增加了一层leakrelu
class GINConv_v2(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 
        

    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, out_dim, aggr = "add", act_func='relu', **kwargs):
        kwargs.setdefault('aggr', aggr)
        self.aggr = aggr
        super(GINConv_v2, self).__init__(**kwargs)
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), nn.BatchNorm1d(2*emb_dim), get_activation(act_func), torch.nn.Linear(2*emb_dim, out_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.add_selfloop = False
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index, edge_attr):
        # #add self loops in the edge space
        if self.add_selfloop:
            edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))
            #add features corresponding to self-loop edges.
            self_loop_attr = torch.zeros(x.size(0), 2)
            self_loop_attr[:,0] = 4 #bond type for self-loop edge
            self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
            edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        # return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return self.activation(x_j + edge_attr)

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GINConv_v3(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 
        

    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, out_dim, aggr = "add", act_func='relu', **kwargs):
        kwargs.setdefault('aggr', aggr)
        self.aggr = aggr
        super(GINConv_v3, self).__init__(**kwargs)
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), nn.BatchNorm1d(2*emb_dim), get_activation(act_func), torch.nn.Linear(2*emb_dim, out_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.add_selfloop = False
        self.activation = nn.PReLU()

    def forward(self, x, edge_index, edge_attr):
        # #add self loops in the edge space
        if self.add_selfloop:
            edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))
            #add features corresponding to self-loop edges.
            self_loop_attr = torch.zeros(x.size(0), 2)
            self_loop_attr[:,0] = 4 #bond type for self-loop edge
            self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
            edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        # return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return self.activation(x_j + edge_attr)

    def update(self, aggr_out):
        return self.mlp(aggr_out)




class GNN(torch.nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """
    def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, emb_dim, aggr = "add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == 'first_cat':
            node_representation = torch.cat([h_list[0], h_list[-1]], dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        return node_representation


# GNN_v2是 GNN + transformer
class GNN_v2(torch.nn.Module):
    """


    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """
    def compress(self,h, temperature=1.0):

        p = self.compressor(h)
        bias = 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(p.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(h.device)
        gate_inputs = (gate_inputs + p) / temperature
        gate_inputs = torch.sigmoid(gate_inputs)
        p = torch.sigmoid(gate_inputs)
        return gate_inputs, p

    def contrastive_loss(self, solute, solvent, tau):

        batch_size, _ = solute.size()
        solute_abs = solute.norm(dim=1)
        solvent_abs = solvent.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', solute, solvent) / torch.einsum('i,j->ij', solute_abs, solvent_abs)
        sim_matrix = torch.exp(sim_matrix / tau)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss

    def __init__(self, num_layer, emb_dim, input_layer=True, JK="last", drop_ratio=0, gnn_type="gin",
                 gnn_activation='relu',
                 d_model=128, trans_layer=0, nhead=4, dim_feedforward=512, transformer_dropout=0,
                 transformer_activation=F.relu, transformer_norm_input=True, custom_trans=False, drop_mask_tokens=False,ib_remask=False,
                 pe_dim=0, trans_pooling='none',graph_pooling="mean"):
        super(GNN_v2, self).__init__()


        self.ib_remask = ib_remask
        self.trans_pooling = trans_pooling
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.drop_mask_tokens = drop_mask_tokens
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        if self.ib_remask:
            self.compressor = nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.BatchNorm1d(emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, 1),
            )


        self.input_layer = input_layer
        if self.input_layer:
            self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
            self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

            torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
            torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = nn.ModuleList()
        self.activations = nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, emb_dim, aggr="add", act_func=gnn_activation))
            elif gnn_type == 'gin_v2':
                self.gnns.append(GINConv_v2(emb_dim, emb_dim, aggr="add", act_func=gnn_activation))
            elif gnn_type == 'gin_v3':
                self.gnns.append(GINConv_v3(emb_dim, emb_dim, aggr="add", act_func=gnn_activation))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))
            else:
                raise NotImplementedError()
            self.activations.append(get_activation(gnn_activation))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        self.trans_layer = trans_layer
        if self.trans_layer > 0:
            self.gnn2trans = nn.Linear(emb_dim + pe_dim, d_model, bias=False)
            self.gnn2trans_act = get_activation(gnn_activation)
            self.trans_enc = TransformerNodeEncoder_v3(d_model, trans_layer, nhead, dim_feedforward,
                                                       transformer_dropout, transformer_activation,
                                                       transformer_norm_input, custom_trans=custom_trans)

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, x, edge_index, edge_attr, batch=None, mask_tokens=None, pe_tokens=None):
        if self.input_layer:
            x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(self.activations[layer](h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == 'first_cat':
            node_representation = torch.cat([h_list[0], h_list[-1]], dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "random":
            random_number = random.randint(self.num_layer-2, self.num_layer)
            node_representation = h_list[random_number]

        ## add pe tokens
        if pe_tokens is not None:
            node_representation = torch.cat((node_representation, pe_tokens), dim=-1)
        preserve_rate=KL_loss=None
        if self.trans_layer > 0:
            assert batch is not None
            if self.ib_remask:
                h_detach = node_representation.detach()
                detach = False
                if detach:

                    node_feature_mean = scatter_mean(h_detach, batch, dim=0)[batch]
                    node_feature_std = scatter_std(h_detach, batch, dim=0)[batch]
                    lambda_pos, p = self.compress(h_detach,temperature=1)
                    # lambda_pos, p = self.compress(node_representation, temperature=1)
                    preserve_rate = p.mean()
                    lambda_neg = 1 - lambda_pos
                    # noisy_node_feature_mean = lambda_pos * node_representation + lambda_neg * node_feature_mean
                    noisy_node_feature_mean = lambda_pos * h_detach + lambda_neg * node_feature_mean
                    noisy_node_feature_std = lambda_neg * node_feature_std
                else:
                    node_feature_mean = scatter_mean(h_detach, batch, dim=0)[batch]
                    node_feature_std = scatter_std(h_detach, batch, dim=0)[batch]
                    # lambda_pos, p = self.compress(h_detach, temperature=1)
                    lambda_pos, p = self.compress(node_representation, temperature=1)
                    preserve_rate = p.mean()
                    lambda_neg = 1 - lambda_pos
                    noisy_node_feature_mean = lambda_pos * node_representation + lambda_neg * node_feature_mean
                    # noisy_node_feature_mean = lambda_pos * h_detach + lambda_neg * node_feature_mean
                    noisy_node_feature_std = lambda_neg * node_feature_std

                noisy_node_feature = noisy_node_feature_mean + torch.rand_like(
                    noisy_node_feature_mean) * noisy_node_feature_std
                epsilon = 1e-7
                KL_tensor = 0.5 * scatter_add(
                    ((noisy_node_feature_std ** 2) / (node_feature_std + epsilon) ** 2).mean(dim=1), batch).reshape(-1,
                                                                                                                    1) + \
                            scatter_add(
                                (((noisy_node_feature_mean - node_feature_mean) / (node_feature_std + epsilon)) ** 2),
                                batch, dim=0)
                KL_Loss = torch.mean(KL_tensor)
                # g = self.pool(h, batch)
                # g_noise = self.pool(noisy_node_feature, batch)
                # cont_loss = self.contrastive_loss(g_noise, g.detach(), 1)
                # cont_loss
            if self.drop_mask_tokens:
                assert mask_tokens is not None
                if self.ib_remask:
                    mask_tokens = torch.zeros(batch.size(0), dtype=torch.bool)
                    mask_index = torch.argsort(lambda_pos.squeeze(),descending=True)[:mask_tokens.sum()]
                    # mask_index = p.squeeze() > 0.5
                    mask_tokens[mask_index] = True
                unmask_tokens = ~mask_tokens
                node_representation = node_representation[unmask_tokens]
                node_representation = self.gnn2trans_act(self.gnn2trans(node_representation))
                pad_x, pad_mask = to_dense_batch(node_representation,
                                                 batch[unmask_tokens])  # shape = [B, N_max, D], shape = [B, N_max]
                pad_x = pad_x.permute(1, 0, 2)
                pad_x, _ = self.trans_enc(pad_x, ~pad_mask)  # discard the cls token; shape = [N_max+1, B, D]
                if self.trans_pooling == 'cls':
                    return pad_x[-1]
                pad_x = pad_x[:-1]  # discard the cls token; shape = [N_max, B, D]
                node_representation = pad_x.permute(1, 0, 2)[pad_mask]
            else:
                if self.ib_remask:
                    node_representation = noisy_node_feature
                node_representation = self.gnn2trans_act(self.gnn2trans(node_representation))
                pad_x, pad_mask = to_dense_batch(node_representation,
                                                 batch)  # shape = [B, N_max, D], shape = [B, N_max]
                pad_x = pad_x.permute(1, 0, 2)
                pad_x, _ = self.trans_enc(pad_x, ~pad_mask)  # discard the cls token; shape = [N_max+1, B, D]
                if self.trans_pooling == 'cls':
                    return pad_x[-1]
                pad_x = pad_x[:-1]  # discard the cls token; shape = [N_max, B, D]
                node_representation = pad_x.permute(1, 0, 2)[pad_mask]
        if self.drop_mask_tokens and self.ib_remask:
            return node_representation,KL_Loss,mask_tokens,preserve_rate
        if not self.drop_mask_tokens and self.ib_remask:

            return node_representation,KL_Loss,mask_tokens,preserve_rate
        return node_representation




class GNNDecoder_v2(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, gnn_layer=1, drop_ratio=0, gnn_type="gin", gnn_activation='relu',
                 gnn_jk='last',
                 d_model=128, trans_layer=0, nhead=4, dim_feedforward=512, transformer_dropout=0,
                 transformer_activation=F.relu, transformer_norm_input=True, custom_trans=False, drop_mask_tokens=False,
                 pe_dim=0, use_input_norm=False, zero_mask=False):
        super().__init__()
        self.gnn_jk = gnn_jk
        self.num_layer = gnn_layer
        self.drop_mask_tokens = drop_mask_tokens
        self.gnns = nn.ModuleList()
        self.activations = nn.ModuleList()
        for layer in range(gnn_layer - 1):
            if gnn_type == "gin":
                self.gnns.append(GINConv(hidden_dim, hidden_dim, aggr="add", act_func=gnn_activation))
            elif gnn_type == 'gin_v2':
                self.gnns.append(GINConv_v2(hidden_dim, hidden_dim, aggr="add", act_func=gnn_activation))
            elif gnn_type == 'gin_v3':
                self.gnns.append(GINConv_v3(hidden_dim, hidden_dim, aggr="add", act_func=gnn_activation))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(hidden_dim, hidden_dim, aggr="add"))
            elif gnn_type == "linear":
                self.gnns.append(nn.Linear(hidden_dim, hidden_dim))
            else:
                raise NotImplementedError(f"{gnn_type}")
            self.activations.append(get_activation(gnn_activation))

        if trans_layer > 0:
            next_dim = hidden_dim
        else:
            if gnn_jk == 'concat':
                self.combine = nn.Linear(hidden_dim * gnn_layer, out_dim)
                next_dim = hidden_dim
            elif gnn_jk == 'last':
                next_dim = out_dim
            else:
                raise NotImplementedError()

        if gnn_type == "gin":
            self.gnns.append(GINConv(hidden_dim, next_dim, aggr="add", act_func=gnn_activation))
        elif gnn_type == 'gin_v2':
            self.gnns.append(GINConv_v2(hidden_dim, next_dim, aggr="add", act_func=gnn_activation))
        elif gnn_type == 'gin_v3':
            self.gnns.append(GINConv_v3(hidden_dim, next_dim, aggr="add", act_func=gnn_activation))
        elif gnn_type == "gcn":
            self.gnns.append(GCNConv(hidden_dim, next_dim, aggr="add"))
        elif gnn_type == "linear":
            self.gnns.append(nn.Linear(hidden_dim, next_dim))
        else:
            raise NotImplementedError(f"{gnn_type}")
        self.activations.append(get_activation(gnn_activation))

        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(drop_ratio)
        self.enc_to_dec = torch.nn.Linear(in_dim, hidden_dim, bias=False)

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(gnn_layer - 1):
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))
        self.batch_norms.append(torch.nn.BatchNorm1d(next_dim))
        if zero_mask:
            self.mask_embed = nn.Parameter(torch.zeros((1, hidden_dim,)), requires_grad=False)
        else:
            self.mask_embed = nn.Parameter(torch.zeros((1, hidden_dim,)))
            nn.init.normal_(self.mask_embed, std=.02)

        self.trans_layer = trans_layer
        if self.trans_layer > 0:
            if self.gnn_jk == 'last':
                self.gnn2trans = nn.Linear(hidden_dim + pe_dim, d_model, bias=False)
            elif self.gnn_jk == 'concat':
                self.gnn2trans = nn.Linear(hidden_dim * gnn_layer + pe_dim, d_model, bias=True)
            else:
                raise NotImplementedError()
            self.gnn2trans_act = get_activation(gnn_activation)
            self.trans2out = nn.Linear(d_model, out_dim, bias=False)
            self.trans_enc = TransformerNodeEncoder_v3(d_model, trans_layer, nhead, dim_feedforward,
                                                       transformer_dropout, transformer_activation,
                                                       transformer_norm_input, custom_trans=custom_trans)

        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            self.input_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, edge_index, edge_attr, masked_tokens, batch, pe_tokens=None):
        x = self.activation(x)
        x = self.enc_to_dec(x)

        if self.drop_mask_tokens:
            ## recover the masked tokens
            box = self.mask_embed.repeat(batch.shape[0], 1)
            # print(box.shape,x.shape,masked_tokens.shape)
            box[~masked_tokens] = x
            x = box
        else:
            ## re-masking
            x = torch.where(masked_tokens.reshape(-1, 1), self.mask_embed, x)

        if self.use_input_norm:
            x = self.input_norm(x)

        xs = []
        for layer in range(self.num_layer):
            x = self.gnns[layer](x, edge_index, edge_attr)
            x = self.batch_norms[layer](x)
            if layer != self.num_layer - 1 or self.gnn_jk == 'concat':
                x = self.activations[layer](x)
            x = self.dropout(x)
            xs.append(x)

        if self.trans_layer > 0:
            if pe_tokens is not None:
                x = torch.cat((x, pe_tokens), dim=-1)
            if self.gnn_jk == 'concat':
                x = torch.cat(xs, dim=-1)

            x = self.gnn2trans_act(self.gnn2trans(x))
            assert batch is not None
            pad_x, pad_mask = to_dense_batch(x, batch)  # shape = [B, N_max, D], shape = [B, N_max]
            pad_x = pad_x.permute(1, 0, 2)
            pad_out, _ = self.trans_enc(pad_x, ~pad_mask)  # discard the cls token; shape = [N_max+1, B, D]
            pad_out = pad_out[:-1]  # discard the cls token; shape = [N_max, B, D]
            trans_out = pad_out.permute(1, 0, 2)[pad_mask]
            trans_out = self.trans2out(trans_out)
            x = trans_out
        else:
            if self.gnn_jk == 'last':
                x = xs[-1]
            elif self.gnn_jk == 'concat':
                x = self.combine(torch.cat(xs, dim=-1))
            else:
                raise NotImplementedError()
        return x

class TokenMAE(torch.nn.Module):
    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("GNNTransformer - Training Config")
        ## gnn parameters
        group.add_argument('--gnn_emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
        group.add_argument('--gnn_dropout', type=float, default=0) # follow the setting of MAE
        group.add_argument('--gnn_JK', type=str, default='last')
        group.add_argument('--gnn_type', type=str, default='gin')
        group.add_argument("--gnn_activation", type=str, default="relu")
        group.add_argument("--decoder_jk", type=str, default="last")

        ## transformer parameters
        group.add_argument('--d_model', type=int, default=128)
        group.add_argument("--dim_feedforward", type=int, default=512, help="transformer feedforward dim")
        group.add_argument("--nhead", type=int, default=4, help="transformer heads")
        group.add_argument("--transformer_dropout", type=float, default=0) # follow the setting of MAE
        group.add_argument("--transformer_activation", type=str, default="relu")
        group.add_argument("--transformer_norm_input", action="store_true", default=True)
        group.add_argument('--custom_trans', action='store_true', default=True)
        group.add_argument('--drop_mask_tokens', action='store_true', default=False)
        group.add_argument('--use_trans_decoder', action='store_true', default=False)
        # group.add_argument("--max_input_len", default=1000, help="The max input length of transformer input")
        
        ## encoder parameters
        group.add_argument('--gnn_token_layer', type=int, default=1)
        group.add_argument('--gnn_encoder_layer', type=int, default=5)
        group.add_argument('--trans_encoder_layer', type=int, default=0)

        ## decoder parameters
        group.add_argument('--gnn_decoder_layer', type=int, default=3)
        group.add_argument('--decoder_input_norm', action='store_true', default=False)
        group.add_argument('--trans_decoder_layer', type=int, default=0)
        
        ## others
        group.add_argument('--nonpara_tokenizer', action='store_true', default=False)
        group.add_argument('--moving_average_decay', type=float, default=0.99)
        group.add_argument('--loss', type=str, default='mse')
        group.add_argument('--loss_all_nodes', action='store_true', default=False)
        group.add_argument('--subgraph_mask', action='store_true', default=False)
        group.add_argument('--zero_mask', action='store_true', default=False)
        group.add_argument('--eps', type=float, default=0.5)

        group_pe = parser.add_argument_group("PE Config")
        group_pe.add_argument('--pe_type', type=str, default='none',choices=['none', 'signnet', 'lap', 'lap_v2', 'signnet_v2', 'rwse', 'signnet_v3'])
        group_pe.add_argument('--laplacian_norm', type=str, default='none')
        group_pe.add_argument('--max_freqs', type=int, default=20)
        group_pe.add_argument('--eigvec_norm', type=str, default='L2')
        group_pe.add_argument('--raw_norm_type', type=str, default='none', choices=['none', 'batchnorm'])
        group_pe.add_argument('--kernel_times', type=list, default=[]) # cmd line param not supported yet
        group_pe.add_argument('--kernel_times_func', type=str, default='none')
        group_pe.add_argument('--layers', type=int, default=3)
        group_pe.add_argument('--post_layers', type=int, default=2)
        group_pe.add_argument('--dim_pe', type=int, default=28, help='dim of node positional encoding')
        group_pe.add_argument('--phi_hidden_dim', type=int, default=32)
        group_pe.add_argument('--phi_out_dim', type=int, default=32)

        group.add_argument('--ib_remask',action='store_true', default=False)

        group.add_argument('--use_atom_token', action='store_true', default=False)


    def __init__(self, gnn_encoder_layer, gnn_token_layer, gnn_decoder_layer, gnn_emb_dim, nonpara_tokenizer=False, gnn_JK = "last", gnn_dropout = 0, gnn_type = "gin",
    d_model=128, trans_encoder_layer=0, trans_decoder_layer=0, nhead=4, dim_feedforward=512, transformer_dropout=0, transformer_activation=F.relu, transformer_norm_input=True, custom_trans=False, drop_mask_tokens=False,
                 use_trans_decoder=False, pe_type='none', graph_pooling="mean",use_atom_token=False,args=None):
        super().__init__()
        # assert gnn_JK == 'last'


        ## forward decoder

        self.ib_remask=args.ib_remask

        self.msn = init_msn_loss(
            num_views=1,
            tau=args.tau,
            me_max=True,
            return_preds=True)

        self.node_prototypes, self.node_proto_labels = get_prototype(args.num_proto,gnn_emb_dim,device=args.device)
        if args.freeze_proto:
            self.node_prototypes.requires_grad_ = False
        args.monent = False
        if args.monent:
            self.moment_tokenizer = copy.deepcopy(self.tokenizer)

        self.args = args
        self.pe_type = pe_type
        self.loss_all_nodes = args.loss_all_nodes
        self.loss = args.loss
        self.pos_encoder = PosEncoder(args)
        
        self.tokenizer = GNN_v2(gnn_token_layer, gnn_emb_dim, True, JK=gnn_JK, drop_ratio=gnn_dropout, gnn_type=gnn_type, gnn_activation=args.gnn_activation)
        self.gnn_act = get_activation(args.gnn_activation)
        self.encoder = GNN_v2(gnn_encoder_layer-gnn_token_layer, gnn_emb_dim, False, JK=gnn_JK, drop_ratio=gnn_dropout, gnn_type=gnn_type, gnn_activation=args.gnn_activation,
        d_model=d_model, trans_layer=trans_encoder_layer, nhead=nhead, dim_feedforward=dim_feedforward, transformer_dropout=transformer_dropout, transformer_activation=transformer_activation, transformer_norm_input=transformer_norm_input, custom_trans=custom_trans, drop_mask_tokens=drop_mask_tokens, pe_dim=self.pos_encoder.pe_dim,ib_remask=self.ib_remask)
        self.nonpara_tokenizer = nonpara_tokenizer

        self.mask_embed = nn.Parameter(torch.zeros(gnn_emb_dim))
        nn.init.normal_(self.mask_embed, std=.02)

        if self.nonpara_tokenizer:
            self.tokenizer_nonpara = Tokenizer(gnn_emb_dim, gnn_token_layer, args.eps, JK=gnn_JK, gnn_type='gin')
            
        if gnn_token_layer == 0:
            out_dim = num_atom_type
        else:
            out_dim = gnn_emb_dim

        if trans_encoder_layer > 0:
            in_dim = d_model
        else:
            in_dim = gnn_emb_dim
        
        self.use_trans_decoder = use_trans_decoder
        if self.use_trans_decoder:
            in_dim = d_model + self.pos_encoder.pe_dim
            self.decoder = TransDecoder(in_dim, out_dim, d_model=d_model, trans_layer=trans_decoder_layer, nhead=nhead, dim_feedforward=dim_feedforward, transformer_dropout=transformer_dropout, transformer_activation=transformer_activation, transformer_norm_input=transformer_norm_input, custom_trans=custom_trans, drop_mask_tokens=drop_mask_tokens)
            # self.decoder = GNNDecoder_v3(in_dim, gnn_emb_dim, gnn_emb_dim, gnn_decoder_layer, gnn_type=gnn_type, 
            # d_model=d_model, trans_layer=trans_decoder_layer, nhead=nhead, dim_feedforward=dim_feedforward, transformer_dropout=transformer_dropout, transformer_activation=transformer_activation, transformer_norm_input=transformer_norm_input, custom_trans=custom_trans, drop_mask_tokens=drop_mask_tokens)
        else:
            self.decoder = GNNDecoder_v2(in_dim, gnn_emb_dim, out_dim, gnn_decoder_layer, gnn_type=gnn_type, gnn_activation=args.gnn_activation, gnn_jk=args.decoder_jk,
            d_model=d_model, trans_layer=trans_decoder_layer, nhead=nhead, dim_feedforward=dim_feedforward, transformer_dropout=transformer_dropout, transformer_activation=transformer_activation, transformer_norm_input=transformer_norm_input, custom_trans=custom_trans, drop_mask_tokens=drop_mask_tokens and trans_encoder_layer > 0, pe_dim=self.pos_encoder.pe_dim, use_input_norm=args.decoder_input_norm, zero_mask=args.zero_mask)
        self.use_atom_token = use_atom_token
        if self.use_atom_token:
            self.emb = nn.Embedding(119,gnn_emb_dim)


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        ## forward tokenizer
        h = self.tokenizer(data.x_masked, edge_index, edge_attr)
        
        ## forward tokenizer target

        if not self.use_atom_token:
            with torch.no_grad():
                if self.nonpara_tokenizer:
                    g_tokens = self.tokenizer_nonpara(x, edge_index, self.tokenizer.x_embedding1).detach()
                else:
                    g_tokens = self.tokenizer(x, edge_index, edge_attr).detach()
        else:
            y = batch.x[batch.mask_tokens][:,0]
            # g_tokens = self.emb(y)

            g_tokens = self.emb(y).detach()


        # None
        pe_tokens = self.pos_encoder(data)

        # forward encoder
        if self.ib_remask:

            h,KL_Loss,mask_token,rate= self.encoder(self.gnn_act(h), edge_index, edge_attr, data.batch, data.mask_tokens, pe_tokens)
        else:
            h = self.encoder(self.gnn_act(h), edge_index, edge_attr, data.batch, data.mask_tokens, pe_tokens)
        if not self.ib_remask:
            if not self.args.monent:
                if self.use_trans_decoder:
                    g_pred = self.decoder(h, pe_tokens, data.mask_tokens, data.batch)
                else:
                    g_pred = self.decoder(h, edge_index, edge_attr, data.mask_tokens, data.batch, pe_tokens)
            else:
                g_pred = self.decoder(h, edge_index, edge_attr, data.mask_tokens, data.batch, pe_tokens)
        else:
            if not self.args.monent:
                if self.use_trans_decoder:
                    g_pred = self.decoder(h, pe_tokens, mask_token, data.batch)
                else:
                    g_pred = self.decoder(h, edge_index, edge_attr, mask_token, data.batch, pe_tokens)
            else:
                g_pred = self.decoder(h, edge_index, edge_attr, mask_token, data.batch, pe_tokens)

        ## compute loss
        if not self.loss_all_nodes:
            g_pred = g_pred[data.mask_tokens]
            g_tokens = g_tokens[data.mask_tokens]

        if self.loss == 'mse':
            loss = self.mse_loss(g_tokens, g_pred)
        elif self.loss == 'sce':
            loss = self.sce_loss(g_tokens, g_pred)
        elif self.loss == 'msn':
            loss = self.msn_loss(g_tokens, g_pred)
        else:
            raise NotImplementedError()

        if self.ib_remask:
            return loss,KL_Loss,rate
        else:
            return loss

    def msn_loss(self,node_target_views,node_anchor_views):
        node_anchor_views, node_target_views = node_anchor_views.float(), node_target_views.float().detach()
        (ploss, me_max, ent, logs, _) = self.msn(
            T=self.args.T,
            use_sinkhorn=True,
            use_entropy=True,
            anchor_views=node_anchor_views,
            target_views=node_target_views,
            proto_labels=self.node_proto_labels,
            prototypes=self.node_prototypes)
        node_soft_loss = ploss + self.args.memax_weight * me_max + self.args.ent_weight * ent
        return node_soft_loss


    def mse_loss(self, x, y):
        loss = ((x - y) ** 2).mean()
        return loss

    def sce_loss(self, x, y, alpha: float=1):
        x = F.normalize(x, p=2.0, dim=-1) # shape = [N, D]
        y = F.normalize(y, p=2.0, dim=-1) # shape = [N, D]
        loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
        loss = loss.mean()
        return loss

    @torch.no_grad()
    def update_tokenizer(self,momentum_scheduler):
        m = next(momentum_scheduler)
        for current_params, ma_params in zip(self.tokenizer.parameters(), self.moment_tokenizer.parameters()):
            up_weight, old_weight = current_params.data, ma_params.data
            ma_params.data = (1 - m) * up_weight + m * old_weight



class Tokenizer(torch.nn.Module):
    """
    

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """
    def __init__(self, emb_dim, num_layer, eps, JK = "last", gnn_type = "gin"):
        super().__init__()
        self.num_layer = num_layer
        self.JK = JK
        
        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(NonParaGINConv(eps))
            elif gnn_type == "gcn":
                self.gnns.append(NonParaGCNConv(eps))
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim, affine=False))

    def forward(self, x, edge_index, node_embedding):
        if self.num_layer == 0:
            return F.one_hot(x[:, 0], num_classes=num_atom_type).float()
        x = node_embedding(x[:, 0])
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index)
            h = self.batch_norms[layer](h)
            h_list.append(h)

        ### Different implementations of Jk-concat
        # if self.JK == "concat":
        #     g_tokens = torch.cat(h_list, dim = 1)
        # elif self.JK == 'first_cat':
        #     g_tokens = torch.cat([h_list[0], h_list[-1]], dim = 1)
        # elif self.JK == "last":
        #     g_tokens = h_list[-1]
        # elif self.JK == "max":
        #     h_list = [h.unsqueeze_(0) for h in h_list]
        #     g_tokens = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        # elif self.JK == "sum":
        #     h_list = [h.unsqueeze_(0) for h in h_list]
        #     g_tokens = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]
        g_tokens = h_list[-1]
        return g_tokens


def get_activation(name):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'elu':
        return nn.ELU()
    elif name == 'prelu':
        return nn.PReLU()
    elif name == 'leakyrelu':
        return nn.LeakyReLU()
    else:
        raise NotImplementedError()





# class GNNDecoder_v3(torch.nn.Module):
#     def __init__(self, in_dim, hidden_dim, out_dim, gnn_layer=1, drop_ratio = 0, gnn_type = "gin",
#     d_model=128, trans_layer=0, nhead=4, dim_feedforward=512, transformer_dropout=0, transformer_activation=F.relu, transformer_norm_input=True, custom_trans=False, drop_mask_tokens=False):
#         super().__init__()
#         assert hidden_dim == out_dim
#         self.num_layer = gnn_layer
#         self.drop_mask_tokens = drop_mask_tokens
#         self.gnns = torch.nn.ModuleList()
#         for layer in range(gnn_layer-1):
#             if gnn_type == "gin":
#                 self.gnns.append(GINConv(hidden_dim, hidden_dim, aggr = "add"))
#             elif gnn_type == "gcn":
#                 self.gnns.append(GCNConv(hidden_dim, hidden_dim, aggr = "add"))
#             elif gnn_type == "linear":
#                 self.gnns.append(nn.Linear(hidden_dim, hidden_dim))
#             else:
#                 raise NotImplementedError(f"{gnn_type}")
#
#         if gnn_type == "gin":
#             self.gnns.append(GINConv(hidden_dim, out_dim, aggr = "add"))
#         elif gnn_type == "gcn":
#             self.gnns.append(GCNConv(hidden_dim, out_dim, aggr = "add"))
#         elif gnn_type == "linear":
#             self.gnns.append(nn.Linear(hidden_dim, out_dim))
#         else:
#             raise NotImplementedError(f"{gnn_type}")
#
#         self.activation = nn.PReLU()
#         self.dropout = nn.Dropout(drop_ratio)
#         self.enc_to_dec = torch.nn.Linear(in_dim, hidden_dim, bias=False)
#
#         ###List of batchnorms
#         self.batch_norms = torch.nn.ModuleList()
#         for layer in range(gnn_layer-1):
#             self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))
#         self.batch_norms.append(torch.nn.BatchNorm1d(out_dim))
#         self.use_mask_emb = True
#         self.mask_embed = nn.Parameter(torch.zeros((1, hidden_dim,)))
#         nn.init.normal_(self.mask_embed, std=.02)
#
#         self.trans_layer = trans_layer
#         if self.trans_layer > 0:
#             self.gnn2trans = nn.Linear(hidden_dim, d_model, bias=False)
#             self.trans2out = nn.Linear(d_model, out_dim, bias=False)
#             self.trans_decoder = TransformerNodeDecoder(d_model, trans_layer, nhead, dim_feedforward, transformer_dropout, transformer_activation, transformer_norm_input, custom_trans=custom_trans)
#             self.memory2decoder = nn.Linear(hidden_dim, d_model, bias=False)
#
#     def forward(self, x, edge_index, edge_attr, masked_tokens, batch):
#         x = self.activation(x)
#         x = self.enc_to_dec(x)
#
#         unmask_tokens = ~masked_tokens
#         if self.use_mask_emb:
#             # x[mask_node_indices] = self.mask_embed
#             if self.drop_mask_tokens:
#                 ## get memory
#                 memory = x
#                 memory_batch = batch[unmask_tokens]
#
#                 ## recover the masked tokens
#                 box = self.mask_embed.repeat(batch.shape[0], 1)
#                 box[~masked_tokens] = x
#                 x = box
#             else:
#                 ## get memory
#                 memory = x[unmask_tokens]
#                 memory_batch = batch[unmask_tokens]
#
#                 ## re-mask
#                 x = torch.where(masked_tokens.reshape(-1, 1), self.mask_embed, x)
#         else:
#             ## get memory
#             memory = x[unmask_tokens]
#             memory_batch = batch[unmask_tokens]
#
#             ## re-mask
#             x[masked_tokens] = 0
#
#         for layer in range(self.num_layer):
#             x = self.gnns[layer](x, edge_index, edge_attr)
#             x = self.batch_norms[layer](x)
#             if layer != self.num_layer - 1:
#                 x = F.relu(x)
#             x = self.dropout(x)
#
#         if self.trans_layer > 0:
#             x = F.relu(self.gnn2trans(x))
#             memory = self.memory2decoder(memory)
#             assert batch is not None
#             pad_x, pad_mask = to_dense_batch(x, batch) # shape = [B, N_max, D], shape = [B, N_max]
#             pad_memory, pad_memory_mask = to_dense_batch(memory, memory_batch) # shape = [B, N_max, D], shape = [B, N_max]
#
#             pad_x = pad_x.permute(1, 0, 2)
#             pad_memory = pad_memory.permute(1, 0, 2)
#
#             pad_out = self.trans_decoder(pad_x, pad_memory, ~pad_mask, ~pad_memory_mask) # discard the cls token; shape = [N_max+1, B, D]
#             trans_out = pad_out.permute(1, 0, 2)[pad_mask]
#             trans_out = self.trans2out(trans_out)
#             x = trans_out
#
#         return x


class TransDecoder(torch.nn.Module):
    def __init__(self, in_dim, out_dim, d_model=128, trans_layer=2, nhead=4, dim_feedforward=512, transformer_dropout=0, transformer_activation=F.relu, transformer_norm_input=True, custom_trans=False, drop_mask_tokens=False):
        super().__init__()
        assert trans_layer > 0
        assert drop_mask_tokens

        self.activation = nn.PReLU()
        self.enc_to_dec = torch.nn.Linear(in_dim, d_model)
        ###List of batchnorms
        self.mask_embed = nn.Parameter(torch.zeros((1, d_model)))
        nn.init.normal_(self.mask_embed, std=.02)
        
        self.trans2out = nn.Linear(d_model, out_dim, bias=False)
        self.trans_decoder = TransformerNodeEncoder_v3(d_model, trans_layer, nhead, dim_feedforward, transformer_dropout, transformer_activation, transformer_norm_input, custom_trans=custom_trans)

    def forward(self, x, pos_enc, masked_tokens, batch):
        '''
        x: shape = 
        '''
        ## recover masked nodes
        box = self.mask_embed.repeat(batch.shape[0], 1)
        box[~masked_tokens] = x
        x = box
        
        ## cat pos_enc
        x = torch.cat((x, pos_enc), dim=-1) # shape = [N, d_model + pe_dim]
        x = self.enc_to_dec(x)
        x = self.activation(x)
        
        ## forward transformer encoder for decoding
        pad_x, pad_mask = to_dense_batch(x, batch) # shape = [B, N_max, D], shape = [B, N_max]
        pad_x = pad_x.permute(1, 0, 2)
        pad_out, _ = self.trans_decoder(pad_x, ~pad_mask) # discard the cls token; shape = [N_max+1, B, D]
        pad_out = pad_out[:-1] # discard the cls token; shape = [N_max, B, D]
        trans_out = pad_out.permute(1, 0, 2)[pad_mask]
        trans_out = self.trans2out(trans_out)
        return trans_out


class MaskGNN(torch.nn.Module):
    """
    

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """
    def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super(MaskGNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, emb_dim, aggr = "add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        self.mask_embed = nn.Parameter(torch.zeros(emb_dim))
        nn.init.normal_(self.mask_embed, std=.02)


    #def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        h_list = [x]

        ## the first layer
        h = self.gnns[0](h_list[0], edge_index, edge_attr)
        h = self.batch_norms[0](h)
        
        ## get the target graph tokens
        g_tokens = h.detach()

        
        h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
        
        # conduct masking
        h = torch.where(data.mask_tokens.reshape(-1, 1), self.mask_embed.reshape(1, -1), h)

        h_list.append(h)

        ##  the rest layers
        for layer in range(1, self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == 'first_cat':
            node_representation = torch.cat([h_list[0], h_list[-1]], dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        return node_representation, g_tokens




class GNNDecoder(torch.nn.Module):
    def __init__(self, hidden_dim, out_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super().__init__()
        self._dec_type = gnn_type 
        if gnn_type == "gin":
            self.conv = GINConv(hidden_dim, out_dim, aggr = "add")
        elif gnn_type == "gcn":
            self.conv = GCNConv(hidden_dim, out_dim, aggr = "add")
        elif gnn_type == "linear":
            self.dec = torch.nn.Linear(hidden_dim, out_dim)
        else:
            raise NotImplementedError(f"{gnn_type}")
        self.enc_to_dec = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)    
        self.activation = torch.nn.PReLU() 


    def forward(self, x, edge_index, edge_attr, mask_node_indices):
        if self._dec_type == "linear":
            out = self.dec(x)
        else:
            x = self.activation(x)
            x = self.enc_to_dec(x)
            x[mask_node_indices] = 0
            out = self.conv(x, edge_index, edge_attr)
        return out


class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat
        
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """
    def __init__(self, num_layer, emb_dim, num_tasks, JK = "last", drop_ratio = 0, graph_pooling = "mean", gnn_type = "gin"):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)

        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1
        
        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, self.num_tasks)

    def from_pretrained(self, model_file):
        #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn.load_state_dict(torch.load(model_file))

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)

        return self.graph_pred_linear(self.pool(node_representation, batch))


if __name__ == "__main__":
    pass

