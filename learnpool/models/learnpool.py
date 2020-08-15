import torch
import torch.nn.functional as func
import scipy.sparse as sp
from torch_geometric.nn import SplineConv
from torch_sparse import coalesce


def dense_to_sparse(tensor):
    """
        Computes the sparse matrix of dense matrix
        Adapted from PyTorch Geometric torch_geometric.utils.dense_to_sparse
        Source: https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/utils/sparse.py#L1
    """

    index = tensor.nonzero()
    value = tensor.view(tensor.size(0) * tensor.size(0))

    index = index.t().contiguous()
    index, value = coalesce(index, value, tensor.size(0), tensor.size(1))
    return index, value


def pros_data(data):
    """
        A function to process data used for graph convolution layers
        Args:
        data --> the data object containing all the brain surface information
            data._x --> the input node features of the graph. tensor[nds, 5]
            data._edge_idx --> weighted connectivity indexes. tensor[2, num_edges]
            data._edge_wht --> weight between the edges. tensor[num_edges, 1]
        Returns:
        x --> the input node features of the graph. tensor[nds, 5]
        edge_index --> weighted connectivity indexes. tensor[2, num_edges]
        pseudo_cord --> pseudo-coordinates capturing node relations. tensor[num_edges, 3] 
        ori_adj --> sparse adjaceny matrix. sparsetensor[nds, nds] 
        lapl_gp --> sparse graph laplacian matrix. sparsetensor[nds, nds]
        domain --> aligned spectral coordinates. tensor[nds, 3]
    """
    
    x, edge_index, edge_attr = data._x, data._edge_idx, data._edge_wht.squeeze()
    num_nodes = x.size(0)
    domain = data._x[:, :3]  # data._x = Aligned spectral coordinates[nds, :3] + surface features[nds, 3:]
    # Compute normalized psueudo coordinates
    pseudo_cord = domain[edge_index[0, :], :] - domain[edge_index[1, :], :]  # dist. u_i - u_j
    max_value = pseudo_cord.abs().max()
    pseudo_cord = pseudo_cord / (2 * max_value) + 0.5

    # Compute sparse adj and sparse Laplacian matrix
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
        Learnable pooling operation.         
        Output of the final convolution block (indicated by 'notlast') is given to linear layer. Hence,
        The regularization loss, output adjaceny, and pseudo-coordingates are set to 0
        
        Args:
        mat_y --> computed latent features for each node. tensor[nds, fou1] 
        mat_s --> computed node clusters features used to aggregated the nodes. tensor[nds, clus]
        ori_adj --> sparse adjaceny matrix. sparsetensor[nds, nds]
        lapl_gp --> sparse laplacian matrix. sparsetensor[nds, nds]
        domain --> aligned spectral coordinates. tensor[nds, 3]
        notlast --> 1 - for last graph convolution block; 0 - otherwise. scalar
        Returns:
        out --> pooled  features. tensor[clus, fou1]
        out_adj --> adjaceny matrix for the downsampled graph. tensor[clus, clus]
        reg_lss --> laplacian regularizer. scalar
        out_pseudo_cord --> psueudo coordinates of cluster 
        idx --> connectivity indexes. tensor[2, out_adj_edges]
        val --> weight between the edges. tensor[out_adj_edges, 1]
    """
    
    # Computing the probaiblity of a node belonging to a cluster
    mat_s = torch.softmax(mat_s, dim=-1)
    mat_s_t = mat_s.transpose(0, 1)
    
    # Pooled  features: Y = S^T x Y 
    out = torch.matmul(mat_s_t, mat_y)
    
    # Adjaceny matrix for the downsampled graph: A = S^T x A x S 
    out_adj = torch.matmul(mat_s_t, torch.matmul(ori_adj, mat_s))

    # If intermediate pooling, adj matrix is computed
    if notlast:
        # Laplacian regluarization: L = trace(S^T x L x S)
        reg_lss = torch.trace(torch.matmul(mat_s_t, torch.matmul(lapl_gp, mat_s))) / mat_s.shape[0]
        
        # Compute spectral coordinates and normalized psueudo coordinates of cluster: U = S^T x U 
        out_domain = torch.matmul(mat_s_t, domain)
        out_pseudo_cord = out_domain[idx[0, :], :] - out_domain[idx[1, :], :]
        max_value = out_pseudo_cord.abs().max()
        out_pseudo_cord = out_pseudo_cord / (2 * max_value) + 0.5
        
        # Compute sparse adj matrix
        idx, val = dense_to_sparse(out_adj)

    # Regularization loss, output adjaceny, and pseudo-coordingates are set to 0
    else:
        reg_lss = out_pseudo_cord = idx = val = 0

    return out, out_adj, reg_lss, out_pseudo_cord, idx, val


class GCNet(torch.nn.Module):
    def __init__(self, fin, fou1, clus, fou2, hlin, outp, psudim):
        super(GCNet, self).__init__()
        """
            The Graph convolution block architecture: Set in config file
            fin    --> Input node features
            fou1   --> Output node features for first GC block
            clus   --> Number of clusters learned for first GC block
            fou2   --> Output node features for second GC block
            hlin   --> Output of the first liner layer
            outp   --> Number of output classes
            psudim --> Dimension of the pseudo-coordinates
        """

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
