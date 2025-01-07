import os.path
import torch
import scanpy as sc
import cv2
import squidpy as sq
import anndata
import torch.nn.functional as F
import pandas as pd
import numpy as np


def read_DLPFC(server, inx):
    print('reading anndata...')

    adata = sc.read_visium(path='/.../matrix/{}/'.format(inx),
                               count_file='{}_filtered_feature_bc_matrix.h5'.format(inx))
    print('reading data...')
    label_pth = '/.../truth/{}_truth.txt'.format(inx)
    # read label
    with open(label_pth, 'r') as file:
        lines = file.readlines()
    import pandas as pd
    domain_arr = []
    for line in lines:
        elements = line.strip().split()
        if len(elements) == 1:
            domain_arr.append([elements[0], -1])
        else:
            domain_arr.append([elements[0], elements[1]])

    np_arr = np.array(domain_arr)
    label_match_df = pd.DataFrame(np_arr)
    domain_dict = {'WM': '7', 'Layer_1': '1', 'Layer_2': '2', 'Layer_3': '3',
                   'Layer_4': '4', 'Layer_5': '5', 'Layer_6': '6'}
    label_match_df[1] = label_match_df[1].map(domain_dict)
    labels = label_match_df[1].values

    adata.obs['layer_guess_reordered'] = labels
    adata.obs['layer_guess_reordered'] = adata.obs['layer_guess_reordered'].astype('str').astype('category')
    return adata


def read_mob(server):
    print('reading anndata...')

    adata = sc.read_h5ad(filename='/.../mob/Slide-seqV2_MoB.h5ad')
    spatial_locs = adata.obsm['spatial']
    return adata, spatial_locs


def read_brain():
    print('reading anndata...')
    adata = sc.read_h5ad(filename='/.../brain/MP1.h5ad')
    img_pth = '/.../brain/rotate_3.tif'

    x = torch.from_numpy(adata.obs['x4'].to_numpy()).unsqueeze(-1)
    y = torch.from_numpy(adata.obs['x5'].to_numpy()).unsqueeze(-1)
    adata.obsm['spatial'] = torch.cat([x, y], dim=1).numpy()

    spatial_locs = adata.obsm['spatial']

    # print('calculating image features...')
    img = cv2.imread(img_pth)
    im_container = sq.im.ImageContainer(img)
    spatial_key = "spatial"
    library_id = "tissue42"
    key_added = 'histology'
    adata.uns[spatial_key] = {library_id: {}}
    adata.uns[spatial_key][library_id]["images"] = {}
    adata.uns[spatial_key][library_id]["images"] = {"hires": img}
    adata.uns[spatial_key][library_id]["scalefactors"] = {
        "tissue_hires_scalef": 1,
        "spot_diameter_fullres": 130,
    }
    sq.im.calculate_image_features(adata,
                                   im_container,
                                   features='summary',
                                   key_added=key_added,
                                   n_jobs=4)
    adata.obsm[key_added] = F.normalize(torch.tensor(adata.obsm[key_added].values, dtype=torch.float32),
                                        p=2,
                                        dim=1)

    return adata


def read_pdac(server):
    if server == 'hpc':
        file = '/.../pdac/pdac.txt'
        img_pth = '/.../pdac/pdac.jpg'
        # i_graph_pth = '/.../igraph_pdac_knn.txt'
    if not os.path.exists(file) or not os.path.exists(img_pth):
        raise NameError("no valid ST data file!")

    img = cv2.imread(img_pth)
    # constructing my adata
    with open(file=file, mode='r') as f:
        lines = f.readlines()

    spots = lines[0]
    spots = spots.split()[1:]
    # handle spatial locs
    spatial_locs = []
    for spot in spots:
        loc = spot.split('x')
        # (224, 2)
        spatial_locs.append(loc)
    spatial_locs = np.array(spatial_locs, dtype=np.float32)
    # handle features
    features = []
    genes = []
    raw_features = lines[1:]

    for feature in raw_features:
        arr = feature.split()
        gene = arr[0]
        arr = arr[1:]
        features.append(arr)
        genes.append(gene)
    features = np.array(features, dtype=np.float32).transpose()
    obs = pd.DataFrame(index=spots)
    var = pd.DataFrame(index=genes, columns=['genename'])
    var.iloc[:, 0] = genes
    adata = anndata.AnnData(X=features, obs=obs, var=var)
    spatial_locs *= 378
    spatial_locs[:, 0] += 2700
    spatial_locs[:, 1] += 3300
    adata.obsm['spatial'] = spatial_locs

    # histology
    print('calculating image features...')
    im_container = sq.im.ImageContainer(img)
    spatial_key = "spatial"
    library_id = "tissue42"
    key_added = 'histology'
    adata.uns[spatial_key] = {library_id: {}}
    adata.uns[spatial_key][library_id]["images"] = {}
    adata.uns[spatial_key][library_id]["images"] = {"hires": img}
    adata.uns[spatial_key][library_id]["scalefactors"] = {
        "tissue_hires_scalef": 1,
        "spot_diameter_fullres": 150,
    }

    sq.im.calculate_image_features(adata,
                                   im_container,
                                   features='summary',
                                   key_added=key_added,
                                   n_jobs=4,
                                   scale=1.0)
    adata.obsm[key_added] = F.normalize(torch.tensor(adata.obsm[key_added].values, dtype=torch.float32),
                                        p=2,
                                        dim=1)

    return adata



def read_mer(server, bregma):

    if server == 'bio':
        data_pth = '.../mer/{}_data.csv'.format(bregma)
    else:
        data_pth = '.../mer/{}_data.csv'.format(bregma)
    data = pd.read_csv(data_pth)
    gene_exp = data.values[:, 10:-1].astype(np.float32)
    # gene_exp = np.concatenate([gene_exp[:, :144], gene_exp[:, 145:]], axis=1)
    gene_exp[:, 144] = np.zeros([len(gene_exp)])
    labels = data['GT_label'].values
    gene_exp
    spatial_locs = data[['Centroid_X', 'Centroid_Y']].values
    cells = data['Cell_ID'].values
    genes = data.columns.values[10:-1]

    adata = anndata.AnnData(X=gene_exp,
                            obs=pd.DataFrame(index=cells),
                            var=pd.DataFrame(index=genes),
                            )
    adata.obsm['spatial'] = spatial_locs
    adata.obs['GT_labels'] = labels
    return adata, spatial_locs
