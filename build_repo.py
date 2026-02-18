import argparse
import os
from scipy.spatial import KDTree
import random
import utils
import pickle
import h5py
import gzip


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build repository: Train PCA -> Project features -> Build k-d trees"
    )
    parser.add_argument('--thumb_file', type=str, required=True,
        help='Path to thumbnail features (.h5)')
    parser.add_argument('--fv_file', type=str, required=True,
        help='Path to fisher vector features (.h5)')
    parser.add_argument('--vgg_file', type=str, required=True,
        help='Path to VGG features (.h5)')
    parser.add_argument('--index_file', type=str, required=True,
        help='Path to index file (.pickle)')
    parser.add_argument("--out_path", type=str, required=True,
        help="Path to output folder")
    parser.add_argument("--pca_train_samp", type=int, required=True,
        help="Number of samples to use for PCA training." \
        "150000 for FIVR-200K and 200000 for VCSL Dataset (for our paper).")
    parser.add_argument("--comps", type=int, required=True, nargs='+',
        help="Number of PCA components." \
        "Pass '32 64 128' for thumbnail components=32, fisher vector components=64, and VGG components=128.")
    parser.add_argument("--leafsize", default=32, type=int,
        help="Leafsize for KD-Tree")

    args = parser.parse_args()

    thumb_file = args.thumb_file
    fv_file = args.fv_file
    vgg_file = args.vgg_file
    index_file = args.index_file
    out_path = args.out_path
    pca_train_samples = args.pca_train_samp # 200000 # 150000
    num_of_pca_comp_thumb = args.comps[0]
    num_of_pca_comp_fv = args.comps[1]
    num_of_pca_comp_vgg = args.comps[2]
    leafsize = args.leafsize

    os.makedirs(out_path, exist_ok=True)

    name_suffix = f'_{num_of_pca_comp_fv}_{num_of_pca_comp_thumb}_{num_of_pca_comp_vgg}'

    indx_info_dict = pickle.load(gzip.open(index_file, 'rb'))
    frame_ids = list(range(0, len(indx_info_dict), 1))

    print('--------------------------------------Loading feature array for train--------------------------------------')
    feats_fv = h5py.File(fv_file, 'r')
    feats_thumb = h5py.File(thumb_file, 'r')
    feats_vgg = h5py.File(vgg_file, 'r')

    print('--------------------------------------Training PCA--------------------------------------')
    pca_indxs = random.choices(frame_ids, k=pca_train_samples)
    print('Fisher vector features processing')
    pca_fv = utils.train_PCA_hdf5(feats_fv, pca_indxs, num_of_pca_comp_fv)
    print('Thumbnail features processing')
    pca_thumb = utils.train_PCA_hdf5(feats_thumb, pca_indxs, num_of_pca_comp_thumb)
    print('VGG features processing')
    pca_vgg = utils.train_PCA_hdf5(feats_vgg, pca_indxs, num_of_pca_comp_vgg)
    pca_dict = {
        'pca_fv': pca_fv,
        'pca_thumb': pca_thumb,
        'pca_vgg': pca_vgg,
        'num_of_pca_comp_fv': num_of_pca_comp_fv,
        'num_of_pca_comp_thumb': num_of_pca_comp_thumb,
        'num_of_pca_comp_vgg': num_of_pca_comp_vgg
        }
    pickle.dump(pca_dict, open(f'{out_path}/pca_models{name_suffix}.pkl', 'wb'))
    print (f'Saved PCA models to {out_path}/pca_models{name_suffix}.pkl')

    print('--------------------------------------Projecting features--------------------------------------')
    print('Projecting fisher vector features via trained PCA model')
    pca_features_fv, _ = utils.project_via_PCA_hdf5(feats_fv, pca_fv, frame_ids, f'{out_path}/fv_repo{name_suffix}.npy')
    print(f'Saved projected features to {out_path}/fv_repo{name_suffix}.npy')

    print('Projecting thumbnail features via trained PCA model')
    pca_features_thumb, _ = utils.project_via_PCA_hdf5(feats_thumb, pca_thumb, frame_ids, f'{out_path}/thumb_repo{name_suffix}.npy')
    print(f'Saved projected features to {out_path}/thumb_repo{name_suffix}.npy')

    print('Projecting VGG features via trained PCA model')
    pca_features_vgg, _ = utils.project_via_PCA_hdf5(feats_vgg, pca_vgg, frame_ids, f'{out_path}/vgg_repo{name_suffix}.npy')
    print(f'Saved projected features to {out_path}/vgg_repo{name_suffix}.npy')


    print('--------------------------------------Generating KDTree--------------------------------------')
    tree_thumb = KDTree(pca_features_thumb.squeeze(axis=1), leafsize=leafsize)
    tree_fv = KDTree(pca_features_fv.squeeze(axis=1), leafsize=leafsize)
    tree_vgg = KDTree(pca_features_vgg.squeeze(axis=1), leafsize=leafsize)
    save_dict = {
        'tree_thumb': tree_thumb,
        'tree_fv': tree_fv,
        'tree_vgg': tree_vgg,
        'leafsize': leafsize
        }
    save_dict_file = open(f'{out_path}/tree_data{name_suffix}.pkl', 'wb')
    pickle.dump(save_dict, save_dict_file)
    print (f'Saved k-d trees repository to {out_path}/tree_data{name_suffix}.pkl')

