import argparse
from scipy.spatial import KDTree
import numpy as np
import utils
import time
import pickle
import yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Efficient Frame Retrieval via Multiple k-d Tree Setup")

    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file")
    
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    feature_path_npy_thumb = config["paths"]["thumb_train"]
    feature_path_npy_thumb_test = config["paths"]["thumb_test"]
    feature_path_npy_fv = config["paths"]["fv_train"]
    feature_path_npy_fv_test = config["paths"]["fv_test"]
    feature_path_npy_vgg = config["paths"]["vgg_train"]
    feature_path_npy_vgg_test = config["paths"]["vgg_test"]
    out_path = config["paths"]["out_path"]

    k = config["kdtree"]["k"]
    leafsize = config["kdtree"]["leafsize"]

    num_of_pca_comp_t = config["pca"]["thumb_components"]
    num_of_pca_comp_fv = config["pca"]["fv_components"]
    num_of_pca_comp_vgg = config["pca"]["vgg_components"]
    pca_train_samples = config["pca"]["train_samples"]
    save_pca = config["pca"]["save_pca"]


    print('--------------------------------------Loading feature array for train--------------------------------------')
    kdtree_list_thumb, frame_id_to_name, frame_name_to_id = utils.npy_to_kdtree_list(feature_path_npy_thumb)
    kdtree_list_fv, _, _ = utils.npy_to_kdtree_list(feature_path_npy_fv)
    kdtree_list_vgg, _, _ = utils.npy_to_kdtree_list(feature_path_npy_vgg)
    print('--------------------------------------Loading feature array for test--------------------------------------')
    kdtree_list_thumb_test, frame_ids_test = utils.npy_to_kdtree_list_test(feature_path_npy_thumb_test, frame_name_to_id)
    kdtree_list_fv_test, _ = utils.npy_to_kdtree_list_test(feature_path_npy_fv_test, frame_name_to_id)
    kdtree_list_vgg_test, _ = utils.npy_to_kdtree_list_test(feature_path_npy_vgg_test, frame_name_to_id)
    print(f'Thumb dim (before dimension reduction) = {kdtree_list_thumb.shape}\n'
            f'FV dim (before dimension reduction) = {kdtree_list_fv.shape}\n'
            f'VGG dim (before dimension reduction) = {kdtree_list_vgg.shape}\n')

    '''Train PCA'''
    print('--------------------------------------Training PCA--------------------------------------')
    pca_thumb = utils.train_PCA(kdtree_list_thumb, num_of_pca_comp_t, pca_train_samples)
    pca_fv = utils.train_PCA(kdtree_list_fv, num_of_pca_comp_fv, pca_train_samples)
    pca_vgg = utils.train_PCA(kdtree_list_vgg, num_of_pca_comp_vgg, pca_train_samples)

    if save_pca == True:
        pickle.dump(pca_thumb, open(f'{out_path}/trained_pca_thumb.pkl', 'wb'))
        pickle.dump(pca_fv, open(f'{out_path}/trained_pca_fv.pkl', 'wb'))
        pickle.dump(pca_vgg, open(f'{out_path}/trained_pca_vgg.pkl', 'wb'))

    print('--------------------------------------Project features with trained PCA model--------------------------------------')
    pca_features_thumb = utils.project_via_PCA(kdtree_list_thumb, pca_thumb)
    pca_features_fv = utils.project_via_PCA(kdtree_list_fv, pca_fv)
    pca_features_vgg = utils.project_via_PCA(kdtree_list_vgg, pca_vgg)
    pca_features_thumb_test = utils.project_via_PCA(kdtree_list_thumb_test, pca_thumb)
    pca_features_fv_test = utils.project_via_PCA(kdtree_list_fv_test, pca_fv)
    pca_features_vgg_test = utils.project_via_PCA(kdtree_list_vgg_test, pca_vgg)
    print(f'Thumb dim (after dimension reduction) = {pca_features_thumb.shape}\n'
        f'FV dim (after dimension reduction) = {pca_features_fv.shape}\n'
        f'VGG dim (after dimension reduction) = {pca_features_vgg.shape}\n')

    print('--------------------------------------Generating KDTree--------------------------------------')
    tree_thumb = KDTree(pca_features_thumb, leafsize=leafsize)
    tree_fv = KDTree(pca_features_fv, leafsize=leafsize)
    tree_vgg = KDTree(pca_features_vgg, leafsize=leafsize)
    save_dict = {
        'tree_thumb': tree_thumb,
        'tree_fv': tree_fv,
        'tree_vgg': tree_vgg,
        'leafsize': leafsize
        }
    save_dict_file = open(f'{out_path}/tree_data.pkl', 'wb')
    pickle.dump(save_dict, save_dict_file)

    print('--------------------------------------Quering KDTrees for test--------------------------------------')
    time1 = time.time()
    _, pred_frame_ids_thumb = tree_thumb.query(pca_features_thumb_test, k=k, workers=-1)
    _, pred_frame_ids_fv = tree_fv.query(pca_features_fv_test, k=k, workers=-1)
    _, pred_frame_ids_vgg = tree_vgg.query(pca_features_vgg_test, k=k, workers=-1)

    gt_frame_ids = frame_ids_test
    final_pred_frame_idsall, final_gt_frame_ids = [], []
    final_pred_frame_ids_thumb, final_pred_frame_ids_fv, final_pred_frame_ids_vgg = [], [], []
    for i in range(len(pred_frame_ids_thumb)):
        gt_frame_id = gt_frame_ids[i]
        pred_frame_ids = np.unique(np.concatenate((pred_frame_ids_thumb[i], pred_frame_ids_fv[i], pred_frame_ids_vgg[i]),0)).tolist()

        final_pred_frame_idsall.append(pred_frame_ids)
        final_gt_frame_ids.append(gt_frame_id)
        final_pred_frame_ids_thumb.append(pred_frame_ids_thumb[i])
        final_pred_frame_ids_fv.append(pred_frame_ids_fv[i])
        final_pred_frame_ids_vgg.append(pred_frame_ids_vgg[i])

        # print('Groundtruth frame ID =', gt_frame_id, '\nPredicted frame ID =',  pred_frame_ids_fv[i])

    time2 = time.time()

    '''Recall calculation'''
    recall_all = utils.calculate_recall(final_gt_frame_ids, final_pred_frame_idsall)
    recall_thumb = utils.calculate_recall(final_gt_frame_ids, final_pred_frame_ids_thumb)
    recall_fv = utils.calculate_recall(final_gt_frame_ids, final_pred_frame_ids_fv)
    recall_vgg = utils.calculate_recall(final_gt_frame_ids, final_pred_frame_ids_vgg)

    recall_res_txt = (f'Total time in seconds = {time2-time1:.4f}\n'
            f'Time per query in msec = {(time2-time1)*1000/len(pred_frame_ids_thumb):.4f}\n'
            f'Thumbnail Feature Recall = {recall_thumb:.4f}\n'
            f'Fisher Vector Feature Recall = {recall_fv:.4f}\n'
            f'VGG Feature Recall = {recall_vgg:.4f}\n'
            f'Total Recall = {recall_all:.4f}')
    
    print (recall_res_txt)
    
    with open(f'{out_path}/frame_retrieval_recall_results.txt', 'w') as f:
        f.write(recall_res_txt)

