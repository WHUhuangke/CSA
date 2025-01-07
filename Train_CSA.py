import torch
import numpy as np
import read_data
import util
from util import normalize_adj, preprocess_data, constr_feature_graph, sparse_mx_to_torch_sparse_tensor
import scipy.sparse as sp
import warnings
from tqdm import tqdm
from CSA import CSA, Encoder
warnings.filterwarnings("ignore")


def train_CSA(adata,
                  nepoch=1000,
                  his_graph_path='./histology_graph.txt',
                  exp_graph_path='./features_graph.txt',
                  histology=False,
                  histology_key='histology',
                  lr=0.001,
                  device=torch.device('cpu'),
                  l2_coef=0.0,
                  n_enc_out=64,
                  # graph weight
                  spa_w=1.,
                  exp_w=0.,
                  his_w=0.,
                  rec_w=10,
                  t0=500,
                  rec_key = 'reconstr_mat',
                  att_hid=16,
                  num_proj_hid=32,
                  key_added='CSA',
                  detect_method='louvain',
                  ):
    """\
            Training graph convolution encoder

            Parameters
            ----------
            adata
                AnnData object of scanpy package.
            histology
                Whether histology information is available.
            histology_key
                The key of histology features stored in adata.obsm.
            att_hid
                Hidden dimension of attention layer.
            num_proj_hid
                Hidden dimension of projection operation.
            n_enc_out
                Output dimension of GCN encoder.
            nepoch
                Number of total epochs in training.
            lr
                Learning rate for AdamOptimizer.
            spa_w
                Weight of spatial mode.
            exp_w
                Weight of gene expression mode.
            his_w
                Weight of histology mode.
            key_added
                The latent embeddings are saved in adata.obsm[key_added].
            t0
                The moment begin to use community strength augmented constractive loss.
            exp_graph_path
                The path of stored expression graph.
            his_graph_path
                The path of stored histology graph.
            device
                See torch.device.

            Returns
            -------
            AnnData
            """
    spatial_locs = adata.obsm['spatial']
    graph_spa = util.constr_spa_graph(spatial_locs)
    # preprocess
    adata = preprocess_data(adata)
    if not isinstance(adata.X, np.ndarray):
        X = torch.from_numpy(adata.X.toarray()).to(torch.float32).to(device)
    else:
        X = torch.from_numpy(adata.X).to(torch.float32).to(device)
    n_nodes = int(X.shape[0])
    feature_dim = int(X.shape[1])
    print('node : {}  dim of X : {}'.format(n_nodes, feature_dim))
    # histology is available
    if histology and histology_key in adata.obsm.columns:
        edge_index_spa = util.adj_to_edge_index(graph_spa).to(device)

        graph_hist = util.constr_feature_graph(features=adata.obsm[histology_key],
                                               graph_path=his_graph_path,
                                               method='knn',
                                               n_nodes=n_nodes)
        edge_index_his = util.adj_to_edge_index(graph_hist).to(device)

        graph_weighted = graph_hist.multiply(his_w) + \
                         graph_spa.multiply(spa_w) + \
                         sp.eye(n_nodes)
        graph_weighted = normalize_adj(graph_weighted)  # coo
        sp_weighted_graph = sparse_mx_to_torch_sparse_tensor(graph_weighted).to(device)

    else:
        edge_index_spa = util.adj_to_edge_index(graph_spa).to(device)
        graph_weighted = graph_spa + sp.eye(n_nodes)
        graph_weighted = normalize_adj(graph_weighted)  # coo
        sp_weighted_graph = sparse_mx_to_torch_sparse_tensor(graph_weighted).to(device)

    # training
    encoder = Encoder(in_channels=feature_dim,
                      out_channels=n_enc_out,
                      activation=torch.nn.PReLU(),
                      k=2).to(device)
    model = CSA(encoder=encoder,
                    n_enc_out=n_enc_out,
                    n_rec=feature_dim,
                    att_hid=att_hid,
                    num_proj_hidden=num_proj_hid,
                    device=device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    edge_cor_spa_1, edge_cor_spa_2, x_spa_1, x_spa_2, node_str_spa = \
        util.community_augmentation(X, edge_index_spa, detect_method)

    if histology and histology_key in adata.obsm.columns:
        edge_cor_his_1, edge_cor_his_2, x_his_1, x_his_2, node_str_his = \
            util.community_augmentation(X, edge_index_his, detect_method)

    for epoch in tqdm(range(1, nepoch + 1)):
        optimizer.zero_grad()
        model.train()
        # forward
        if histology and histology_key in adata.obsm.columns:
            z_spa_cor_1 = model(x_spa_1, edge_cor_spa_1)
            z_spa_cor_2 = model(x_spa_2, edge_cor_spa_2)
            z_his_cor_1 = model(x_his_1, edge_cor_his_1)
            z_his_cor_2 = model(x_his_2, edge_cor_his_2)
            z_spa = model(X, edge_index_spa)
            z_his = model(X, edge_index_his)

            team_loss_spa = model.team_up_loss(z1=z_spa_cor_1,
                                               z2=z_spa_cor_2,
                                               cs=node_str_spa,
                                               current_ep=epoch,
                                               t0=t0,
                                               gamma_max=1.0)

            team_loss_his = model.team_up_loss(z1=z_his_cor_1,
                                               z2=z_his_cor_2,
                                               cs=node_str_his,
                                               current_ep=epoch,
                                               t0=t0,
                                               gamma_max=1.0
                                               )
            reconstr_loss = model.reconstr_loss(z_spa=z_spa,
                                                z_his=z_his,
                                                weighted_adj=sp_weighted_graph,
                                                X=X
                                                )
            loss = team_loss_spa * spa_w + \
                   team_loss_his * his_w + \
                   reconstr_loss * rec_w

        else:
            z_spa_cor_1 = model(x_spa_1, edge_cor_spa_1)
            z_spa_cor_2 = model(x_spa_2, edge_cor_spa_2)
            z_spa = model(X, edge_index_spa)
            t_team_loss = model.team_up_loss(z1=z_spa_cor_1,
                                             z2=z_spa_cor_2,
                                             cs=node_str_spa,
                                             current_ep=epoch,
                                             t0=t0, gamma_max=1.0
                                             )
            recon_loss, z_rec = model.reconstr_loss(z_1=z_spa,
                                       weighted_adj=sp_weighted_graph,
                                       features=X)

            loss = t_team_loss + recon_loss * rec_w

        loss.backward()
        optimizer.step()

        if epoch == 2:
            model.eval()
            with torch.no_grad():
                z_t = model(X, edge_index_spa).detach()
                embedding = z_t.to('cpu').numpy()
                _, z_rec = model.reconstr_loss(z_1=z_t,
                                               weighted_adj=sp_weighted_graph,
                                                features=X)
                adata.obsm[key_added] = embedding
                adata.X = z_rec.detach().numpy()
                adata.write('.../adata_151673.h5ad')


    return adata


# dataset_list = ['151507', '151508', '151509', '151510', '151669', '151670', '151671',
#                 '151672', '151673', '151674', '151675', '151676']
# dataset_list = ['151673']
# for inx in dataset_list:
#     adata = read_data.read_DLPFC(server='bio', inx=inx)
#     train_CSA(adata)
