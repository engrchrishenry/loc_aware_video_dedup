import argparse
import os
import utils
import pickle
import h5py
import gzip


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build test repository: Project test features -> Build k-d trees"
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
    parser.add_argument("--pca_model", type=str, required=True,
        help="Path to trained PCA model")
    parser.add_argument("--comps", type=int, required=True, nargs='+',
        help="Number of components for the trained PCA model." \
        "Pass '32 64 128' for thumbnail components=32, fisher vector components=64, and VGG components=128.")

    args = parser.parse_args()

    num_of_pca_comp_thumb = args.comps[0]
    num_of_pca_comp_fv = args.comps[1]
    num_of_pca_comp_vgg = args.comps[2]
    name_suffix = f'_{num_of_pca_comp_fv}_{num_of_pca_comp_thumb}_{num_of_pca_comp_vgg}'

    thumb_file = args.thumb_file
    fv_file = args.fv_file
    vgg_file = args.vgg_file
    index_file = args.index_file
    out_path = args.out_path
    pca_model = args.pca_model

    os.makedirs(out_path, exist_ok=True)


    indx_info_dict = pickle.load(gzip.open(index_file, 'rb'))
    frame_ids = list(range(0, len(indx_info_dict), 1))

    print('--------------------------------------Loading feature array for test--------------------------------------')
    feats_thumb = h5py.File(thumb_file, 'r')
    feats_fv = h5py.File(fv_file, 'r')
    feats_vgg = h5py.File(vgg_file, 'r')

    print('--------------------------------------Loading PCA model--------------------------------------')
    pca_models = pickle.load(open(pca_model, 'rb'))
    pca_fv = pca_models['pca_fv']
    pca_thumb = pca_models['pca_thumb']
    pca_vgg = pca_models['pca_vgg']

    print('--------------------------------------Projecting features--------------------------------------')

    print('Projecting fisher vector features via trained PCA model')
    _ = utils.project_via_PCA_hdf5(feats_fv, pca_fv, frame_ids, f'{out_path}/fv_repo_test{name_suffix}.npy')
    print(f'Saved projected features to {out_path}/fv_repo_test{name_suffix}.npy')

    print('Projecting thumbnail features via trained PCA model')
    _ = utils.project_via_PCA_hdf5(feats_thumb, pca_thumb, frame_ids, f'{out_path}/thumb_repo_test{name_suffix}.npy')
    print(f'Saved projected features to {out_path}/thumb_repo_test{name_suffix}.npy')

    print('Projecting VGG features via trained PCA model')
    _ = utils.project_via_PCA_hdf5(feats_vgg, pca_vgg, frame_ids, f'{out_path}/vgg_repo_test{name_suffix}.npy')
    print(f'Saved projected features to {out_path}/vgg_repo_test{name_suffix}.npy')

