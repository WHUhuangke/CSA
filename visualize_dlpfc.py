import torch.cuda
# from util import mclust_R
from sklearn.metrics import adjusted_rand_score
import warnings
import util
from read_data import read_DLPFC
import scanpy as sc
import matplotlib as plt
from Train_CSA import train_CSA

warnings.filterwarnings("ignore")
key_added = 'CSA'
slice = '151671'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# adata = read_DLPFC(slice=slice)
slice_list = ['151507', '151508', '151509', '151510',
              '151669', '151670', '151671', '151672',
              '151673', '151674', '151675', '151676']
# dataset_list = ['151510', '151669', '151670', '151671',
#                 '151672', '151673', '151674', '151675', '151676']
for slice in slice_list:

    if slice in ['151669', '151670', '151671', '151672']:
        num_cluster = 5
    else:
        num_cluster = 7
    adata = sc.read_h5ad('D:/csboost/plf/adata/adata_151673.h5ad')
    # adata = read_DLPFC(slice)
    # adata = train_CSA(adata, device=device)
    # visualize
    sc.pp.neighbors(adata, use_rep=key_added)

    # mclust
    util.mclust_R(adata, num_cluster=num_cluster, modelNames='EEE', use_rep=key_added)
    domains = adata.obs['mclust'].cat.codes
    plt.rcParams["figure.figsize"] = (5, 5)
    labels = adata.obs['layer_guess_reordered']
    # adata.obsm['spatial'][:, 0] += 450
    # adata.obsm['spatial'][:, 1] += 300
    # ari
    ari = adjusted_rand_score(labels, domains)
    sc.pl.spatial(adata,
                  color=['mclust', 'layer_guess_reordered'],
                  title=['CSA(ARI = %.3f)' % ari, 'Ground truth'],
                  img_key='hires',
                  spot_size=100)
    # gene expression patterns
    genes = ['CAMK2N1', 'PCP4', 'NEFM']
    adata.var_names_make_unique()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.tl.umap(adata)

for gene in genes:
    sc.pl.umap(adata, cmap='coolwarm', color=gene, title=gene)
sc.pl.umap(adata, color='mclust', title='CSA_UMAP')
