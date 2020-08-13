import torch
import torch.nn.functional as func
import scipy.sparse as sp
from torch_geometric.nn import SplineConv
from torch_sparse import coalesce


def dense_to_sparse(tensor):
    """
        Computes the sparse matrix of dense matrix
        Refer the simpler/better PyTorch Geometric: torch_geometric.utils.dense_to_sparse implementation
    """

    index = tensor.nonzero()
    value = tensor.view(tensor.size(0) * tensor.size(0))

    index = index.t().contiguous()
    index, value = coalesce(index, value, tensor.size(0), tensor.size(1))
    return index, value


def pros_data(data):
    """
        Computes pseudo-coordinates from the data
        pseudo-coordinates: Spectral or Polar or Cartesian domain. default: aligned spectral coordinates
        adj and laplacian matrix: sparse cuda tensors
    """
    x, edge_index, edge_attr = data._x, data._edge_idx, data._edge_wht.squeeze()
    num_nodes = x.size(0)
    domain = data._x[:, :3]  # data._x = Aligned spectral coordinates[nds, :3] + surface features[nds, 3:]
    # Compute normalized psueudo coordinates
    pseudo_cord = domain[edge_index[0, :], :] - domain[edge_index[1, :], :]  # dist. u_i - u_j
    max_value = pseudo_cord.abs().max()
    pseudo_cord = pseudo_cord / (2 * max_value) + 0.5

    # Compute adj and Laplacian matrix
    ori_adj = torch.sparse.FloatTensor(edge_index, edge_attr, torch.Size([num_nodes, num_nodes]))
    lapl = sp.csgraph.laplacian(sp.csr_matrix((edge_attr.cpu(),
                                               edge_index.cpu()),
                                              shape=(num_nodes, num_nodes)),
                                normed=True)
    e1, e2, e3 = sp.find(lapl)
    lapl_idx = torch.cat((torch.LongTensor(e2).unsqueeze(0), torch.LongTensor(e1).unsqueeze(0)), 0).cuda()
    lapl_wht = torch.FloatTensor(e3).cuda()
    lapl_gp = torch.sparse.FloatTensor(lapl_idx, lapl_wht, torch.Size([num_nodes, num_nodes]))

    return x, edge_index, pseudo_cord, ori_adj, lapl_gp, domain


def learn_pool(mat_y, mat_s, ori_adj, lapl_gp, domain, notlast):
    """
        Learnable pooling operation:
        Computes the output node features and adj matrix after the downstream pooling operation
        Computes the Laplacian regularization and pseudo-coordinates for next convolution layers

    """

    mat_s = torch.softmax(mat_s, dim=-1)
    mat_s_t = mat_s.transpose(0, 1)

    out = torch.matmul(mat_s_t, mat_y)
    out_adj = torch.matmul(mat_s_t, torch.matmul(ori_adj, mat_s))

    # If intermediate pooling, adj matrix is computed
    if notlast:
        reg_lss = torch.trace(torch.matmul(mat_s_t, torch.matmul(lapl_gp, mat_s))) / mat_s.shape[0]
        idx, val = dense_to_sparse(out_adj)
        out_domain = torch.matmul(mat_s_t, domain)
        out_pseudo_cord = out_domain[idx[0, :], :] - out_domain[idx[1, :], :]
        max_value = out_pseudo_cord.abs().max()
        out_pseudo_cord = out_pseudo_cord / (2 * max_value) + 0.5

    # For last pooling layer adj matrix not computed, graph collapsed to classification/regression task
    else:
        reg_lss = out_pseudo_cord = idx = val = 0

    return out, out_adj, reg_lss, out_pseudo_cord, idx, val


class GCNet(torch.nn.Module):
    def __init__(self, fin, fou1, clus, fou2, hlin, outp, psudim):
        super(GCNet, self).__init__()

        self.gnn1_pool = SplineConv(fin, clus, dim=psudim, kernel_size=1)
        self.gnn1_embd = SplineConv(fin, fou1, dim=psudim, kernel_size=1)

        self.gnn2_pool = SplineConv(fou1, 1, dim=psudim, kernel_size=1)
        self.gnn2_embd = SplineConv(fou1, fou2, dim=psudim, kernel_size=1)

        self.lin1 = torch.nn.Linear(fou2, hlin)
        self.lin2 = torch.nn.Linear(hlin, outp)

    def forward(self, data):
        # create appropriate datafiles
        x, edge_index, pseudo_cord, ori_adj, lapl_gp, domain = pros_data(data)

        # The first convolution block
        mat_s = self.gnn1_pool(x, edge_index, pseudo_cord)
        mat_y = func.relu(self.gnn1_embd(x, edge_index, pseudo_cord))
        mat_x, adj, reg_lss1, pseudo_cord, idx, val = learn_pool(mat_y, mat_s, ori_adj, lapl_gp, domain, notlast=1)

        # The second/last convolution block
        mat_s = self.gnn2_pool(mat_x, idx, pseudo_cord)
        mat_y = func.relu(self.gnn2_embd(mat_x, idx, pseudo_cord))
        mat_x, adj, reg_lss2, pseudo_cord, idx, val = learn_pool(mat_y, mat_s, adj, lapl_gp, domain, notlast=0)

        # Fully connected dense layers
        mat_x = func.relu(self.lin1(mat_x))
        mat_x = self.lin2(mat_x)

        return func.log_softmax(mat_x[0], dim=-1).unsqueeze(0), reg_lss1+reg_lss2
