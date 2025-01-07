import os.path
from typing import Optional
import rpy2.rinterface_lib.sexp
import scanpy
import scipy.sparse
import torch
import numpy as np
import scanpy as sc
from sklearn.neighbors import kneighbors_graph
import gudhi
import networkx as nx
import scipy.sparse as sp
from cdlib import algorithms
from sklearn.metrics.pairwise import cosine_similarity as cos
from sklearn.neighbors import NearestNeighbors
from typing import Sequence
from cdlib.utils import convert_graph_formats
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

def community_augmentation(expression, edge_index, detect_method):
    device = expression.device
    n_nodes = len(expression)
    data = Data(x=expression, edge_index=edge_index)
    g = to_networkx(data, to_undirected=True)
    print('detecting communities...')
    coms = community_detection(detect_method)(g).communities
    com_str, node_str = community_strength(g, coms)
    com_groups = trans(coms, n_nodes)
    com_sz = [len(com) for com in coms]
    print('{} communities found!'.format(len(com_sz)))
    edge_weight = get_edge_weight(edge_index, com_groups, com_str)
    edge_cor_1 = community_corruption(edge_index, edge_weight, p=0.3).to(device)
    edge_cor_2 = community_corruption(edge_index, edge_weight, p=0.4).to(device)
    x_cor_1 = community_attr_vote(expression, node_str, p=0.1).to(device)
    x_cor_2 = community_attr_vote(expression, node_str, p=0.0).to(device)
    return edge_cor_1, edge_cor_2, x_cor_1, x_cor_2, node_str

def trans(communities: Sequence[Sequence[int]],
          num_nodes: int) -> np.ndarray:
    classes = np.full(num_nodes, -1)
    for i, node_list in enumerate(communities):
        classes[np.asarray(node_list)] = i
    return classes


def community_detection(name):
    algs = {
        # non-overlapping algorithms
        'louvain': algorithms.louvain,
        'combo': algorithms.pycombo,
        'leiden': algorithms.leiden,
        'ilouvain': algorithms.ilouvain,
        'eigenvector': algorithms.eigenvector,
        'girvan_newman': algorithms.girvan_newman,
        # overlapping algorithms
        'demon': algorithms.demon,
        'lemon': algorithms.lemon,
        'lpanni': algorithms.lpanni,
    }
    return algs[name]


def community_strength(graph: nx.Graph,
                       communities: Sequence[Sequence[int]]) -> (np.ndarray, np.ndarray):
    graph = convert_graph_formats(graph, nx.Graph)
    coms = {}
    for cid, com in enumerate(communities):
        for node in com:
            coms[node] = cid
    inc, deg = {}, {}
    links = graph.size(weight="weight")
    assert links > 0, "A graph without link has no communities."
    for node in graph:
        try:
            com = coms[node]
            deg[com] = deg.get(com, 0.0) + graph.degree(node, weight="weight")
            for neighbor, dt in graph[node].items():
                weight = dt.get("weight", 1)
                if coms[neighbor] == com:
                    if neighbor == node:
                        inc[com] = inc.get(com, 0.0) + float(weight)
                    else:
                        inc[com] = inc.get(com, 0.0) + float(weight) / 2.0
        except:
            pass
    com_cs = []
    for idx, com in enumerate(set(coms.values())):
        com_cs.append((inc.get(com, 0.0) / links) - (deg.get(com, 0.0) / (2.0 * links)) ** 2)
    com_cs = np.asarray(com_cs)
    node_cs = np.zeros(graph.number_of_nodes(), dtype=np.float32)
    for i, w in enumerate(com_cs):
        for j in communities[i]:
            node_cs[j] = com_cs[i]
    return com_cs, node_cs


def get_edge_weight(edge_index: torch.Tensor,
                    com: np.ndarray,
                    com_cs: np.ndarray) -> torch.Tensor:
    edge_mod = lambda x: com_cs[x[0]] if x[0] == x[1] else -(float(com_cs[x[0]]) + float(com_cs[x[1]]))
    normalize = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    edge_weight = np.asarray([edge_mod([com[u.item()], com[v.item()]]) for u, v in edge_index.T])
    edge_weight = normalize(edge_weight)
    return torch.from_numpy(edge_weight).to(edge_index.device)


def sparse_to_edge_index(sparse: scipy.sparse.csr_matrix) -> torch.Tensor:
    sparse = sparse.tocoo().astype(np.float32)
    edge_index = torch.from_numpy(
        np.vstack((sparse.row, sparse.col)).astype(np.int64))
    return edge_index


def preprocess_data(adata: scanpy.AnnData = None,
                    n_top_genes: int = 3000,
                    ) -> (scanpy.AnnData, sp.csr_matrix):
    print('preprocessing data...')
    error_msg = "no valid adata object !"
    if not adata:
        print(error_msg)
    adata.var_names_make_unique()
    print('selecting highly variable...')
    sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=n_top_genes)
    print('normalizing...')
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if 'highly_variable' in adata.var.columns:
        adata = adata[:, adata.var['highly_variable']]

    return adata


def constr_spa_graph(spatial_locs: np.ndarray,
                      n_neighbors: int = 15) -> sp.csr_matrix:
    assert int(spatial_locs.shape[1] == 2)
    print('constructing topology graph...')
    knn = kneighbors_graph(spatial_locs, n_neighbors=n_neighbors, mode='distance')
    estimated_graph_cut = knn.sum() / float(knn.count_nonzero())
    spatial_locs_list = spatial_locs.tolist()
    n_node = len(spatial_locs_list)
    # use alpha complex to construct topology graph
    alpha_complex = gudhi.AlphaComplex(points=spatial_locs_list)
    simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=estimated_graph_cut ** 2)
    skeleton = simplex_tree.get_skeleton(1)
    initial_graph = nx.Graph()
    initial_graph.add_nodes_from([i for i in range(n_node)])
    for s in skeleton:
        if len(s[0]) == 2:
            initial_graph.add_edge(s[0][0], s[0][1])

    spa_graph = nx.Graph()
    spa_graph.add_nodes_from(initial_graph)
    spa_graph.add_edges_from(initial_graph.edges)

    # Remove self edges
    for i in range(n_node):
        try:
            spa_graph.remove_edge(i, i)
        except:
            pass
    edge_cnt = len(spa_graph.edges())
    nodes = np.ones(edge_cnt)
    rows = []
    cols = []
    for edge in spa_graph.edges():
        row, col = edge
        rows.append(row)
        cols.append(col)

    spa_graph_sp_mx = sp.csr_matrix((nodes, (rows, cols)), shape=(n_node, n_node), dtype=int)
    # symmetrical adj
    spa_graph_sp_mx += spa_graph_sp_mx.transpose()
    spa_graph_sp_mx.data[spa_graph_sp_mx.data==2] = 1
    return spa_graph_sp_mx

def constr_feature_graph(
        graph_path: str,
        n_nodes: int,
        knn_neighbors: int = 15,
        features: Optional[torch.Tensor] = None,
        method: str = 'cos') -> sp.csr_matrix:
    if os.path.exists(graph_path):
        edge_list = np.genfromtxt(graph_path, dtype=int).transpose()
        edge_cnt = int(edge_list.shape[1])
        nodes = np.ones(edge_cnt)
        feature_graph_sp_mx = sp.csr_matrix((nodes, (edge_list[0], edge_list[1])), shape=(n_nodes, n_nodes))

    else:
        print('constructing feature graph...')
        # fea_array = features.numpy()

        indices = np.empty((n_nodes, knn_neighbors + 1))
        np_features = features.cpu().numpy()
        # np_features = sp_features

        if method == 'knn':
            neighbors = NearestNeighbors(n_neighbors=knn_neighbors + 1, algorithm='kd_tree').fit(np_features)
            _, indices = neighbors.kneighbors(np_features)

        elif method == 'cos':

            dist = cos(np_features)
            # k_neighbors_inds = []
            for i in range(dist.shape[0]):
                ind = np.argpartition(dist[i, :], -(knn_neighbors + 1))[-(knn_neighbors + 1):]
                indices[i] = ind
        else:
            raise AttributeError("No support method for knn!")
        assert len(indices) == int(np_features.shape[0])
        n_nodes = len(indices)

        f = open(graph_path, 'w')

        rows = []
        cols = []
        passtime = 0
        # storing feature graph
        for i, neighbors in enumerate(indices):
            for neighbor in neighbors:
                if neighbor == i:
                    passtime += 1
                    pass
                else:
                    rows.append(i)
                    cols.append(neighbor)
                    f.write('{} {}\n'.format(i, int(neighbor)))

        f.close()
        assert len(rows) == len(cols), 'len of rows should be equals to len of cols!'
        edge_cnt = len(rows)
        nodes = np.ones(edge_cnt)
        feature_graph_sp_mx = sp.csr_matrix((nodes, (rows, cols)), shape=(n_nodes, n_nodes))
    # symmetrical adj
    feature_graph_sp_mx += feature_graph_sp_mx.transpose()
    feature_graph_sp_mx.data[feature_graph_sp_mx.data == 2] = 1
    return feature_graph_sp_mx

def adj_to_edge_index(adj):
    n_nodes = int(adj.shape[0])
    # out = torch.tensor([i for i in range(n_nodes)]).unsqueeze(0)
    # self_loop = torch.cat([out,out],dim=0)
    if isinstance(adj, np.ndarray):
        adj = torch.from_numpy(adj)
        edge_index = adj.nonzero().t()
        # edge_index = torch.cat([edge_index,self_loop],dim=1)

    elif isinstance(adj, sp.csr_matrix):
        adj = adj.tocoo().astype(np.float32)
        edge_index = torch.from_numpy(
            np.vstack((adj.row, adj.col)).astype(np.int64))
        # edge_index = torch.cat([edge_index, self_loop], dim=1)
    else:
        raise TypeError("unsupported adj type !")
    return edge_index


def normalize_adj(adj):
    # normalize adjacency matrix by D^-1/2 * A * D^-1/2,in order to apply to gcn conv

    if not np.array_equal(np.array(adj.todense()).transpose(), np.array(adj.todense())):
        raise AttributeError('adj matrix should be symmetrical!')
    adj = adj.tocoo()
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    # Convert a scipy sparse matrix to a torch sparse tensor.
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def mclust_R(adata: sc.AnnData,
             num_cluster: int,
             modelNames: str = 'EEE',
             random_seed=2023,
             use_rep: str = 'DGI_GCN',
             mclust_pth: str = './pdac/pdac_mclust.txt') -> sc.AnnData:
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri as rrn
    rrn.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rrn.numpy2rpy(adata.obsm[use_rep]), num_cluster, modelNames)
    if isinstance(res, rpy2.rinterface_lib.sexp.NULLType):
        return adata
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

def store_embedding(embedding, umap_pth):
    assert isinstance(embedding, np.ndarray)
    with open(umap_pth, 'w') as file:
        for row in embedding:
            for item in row:
                file.write('{} '.format(item))
            file.write('\n')


def read_embedding(emb_pth):
    if not os.path.exists(emb_pth):
        raise FileNotFoundError('no file')
    emb_list = []
    with open(emb_pth, 'r') as file:
        lines = file.readlines()
    for line in lines:
        emb_item = line.split()
        emb_float = [float(x) for x in emb_item]
        emb_list.append(emb_float)
    return np.array(emb_list)


def community_corruption(edge_index: torch.Tensor,
                         edge_weight: torch.Tensor,
                         p: float,
                         threshold: float = 1.) -> torch.Tensor:
    edge_weight = edge_weight / edge_weight.mean() * (1. - p)
    edge_weight = edge_weight.where(edge_weight > (1. - threshold), torch.ones_like(edge_weight) * (1. - threshold))
    edge_weight = edge_weight.where(edge_weight < 1, torch.ones_like(edge_weight) * 1)
    sel_mask = torch.bernoulli(edge_weight).to(torch.bool)
    return edge_index[:, sel_mask]


def community_attr_vote(feature: torch.Tensor,
                        node_str: np.ndarray,
                        p: float,
                        max_threshold: float = 0.7) -> torch.Tensor:
    x = feature.abs()
    device = feature.device
    w = x.t() @ torch.tensor(node_str).to(device)
    w[torch.nonzero(w == 0)] = w.max()  # for redundant attributes of Cora
    w = w.log()
    w = (w.max() - w) / (w.max() - w.min())
    w = w / w.mean() * p
    w = w.where(w < max_threshold, max_threshold * torch.ones(1).to(device))
    w = w.where(w > 0, torch.zeros(1).to(device))
    drop_mask = torch.bernoulli(w).to(torch.bool)
    feature = feature.clone()
    feature[:, drop_mask] = 0.
    return feature
