import numpy as np
import random
from sklearn.decomposition import PCA
import time
from tqdm import tqdm


def train_PCA(kdtree_list, num_of_pca_comp, pca_train_samples):
    kdtree_list_pca = kdtree_list.copy()
    random.shuffle(kdtree_list_pca)
    kdtree_list_pca = kdtree_list_pca[:pca_train_samples]

    pca_model = PCA(n_components=num_of_pca_comp)
    pca_model.fit(kdtree_list_pca)

    return pca_model


def train_PCA_hdf5(feats, pca_indxs, num_of_pca_comp):
    pca_feats = []
    with tqdm(total=len(pca_indxs), desc="Loading features") as pbar:
        for frame_id in pca_indxs:
            vec = feats[str(frame_id)][:]
            pca_feats.append(vec)
            pbar.update(1)
    print ('Started PCA training')
    pca_model = PCA(n_components=num_of_pca_comp)
    pca_model.fit(pca_feats)
    print ('Completed PCA training')
    return pca_model


def project_via_PCA(kdtree_list, pca_model):
    pca_features = pca_model.transform(kdtree_list)
    return pca_features


def project_via_PCA_hdf5(feats, pca_model, frame_ids, save_path=None):
    pca_features = []
    total_time = 0
    with tqdm(total=len(frame_ids), desc="Projecting features") as pbar:
        for frame_id in frame_ids:
            start_time = time.time()
            vec = feats[str(frame_id)][:].reshape(1, -1)
            pca_feature = pca_model.transform(vec)
            end_time = time.time()
            elapsed_time = end_time - start_time
            total_time += elapsed_time
            pca_features.append(pca_feature)
            pbar.update(1)
    average_time = total_time / len(frame_ids)
    if save_path != None:
        np.save(save_path, np.array(pca_features))
    return np.array(pca_features), average_time


def process_frame_ids(feats, pca_model, frame_ids_chunk, result_queue):
    pca_features_chunk = []
    for frame_id in frame_ids_chunk:
        vec = feats[str(frame_id)][:].reshape(1, -1)
        pca_feature = pca_model.transform(vec)
        pca_features_chunk.append(pca_feature)
    result_queue.put(pca_features_chunk)


def calculate_recall(final_gt_frame_ids, final_pred_frame_ids):
    correct_frame_id = 0
    for index, item in enumerate(final_gt_frame_ids):
        frame_id_temp = item
        # print(frame_id_temp, final_pred_frame_ids[index])
        if frame_id_temp in final_pred_frame_ids[index]:
            correct_frame_id += 1
            # print(frame_id_temp, final_pred_frame_ids[index])
    acc = correct_frame_id * 100 / len(final_gt_frame_ids)
    return acc


def npy_to_kdtree_list(feature_path):
    frame_id_to_name = {}
    frame_name_to_id = {}
    MLP_features_all = np.load(feature_path, allow_pickle=True)
    kdtree_list = MLP_features_all[:, 2]
    kdtree_list = np.concatenate(kdtree_list, 0)
    for i in range(MLP_features_all.shape[0]):
        frame_id, f_name, MLP_feature, _, _, _ = MLP_features_all[i]
        frame_id_to_name[frame_id] = f_name
        frame_name_to_id[f_name] = frame_id
    kdtree_list = kdtree_list.astype('float32')
    return kdtree_list, frame_id_to_name, frame_name_to_id
    

def npy_to_kdtree_list_test(feature_path, frame_name_to_id):
    frame_ids = []
    MLP_features_all = np.load(feature_path, allow_pickle=True)
    kdtree_list = MLP_features_all[:, 2]
    kdtree_list = np.concatenate(kdtree_list, 0)
    for i in range(MLP_features_all.shape[0]):
        f_name = MLP_features_all[i][1]
        frame_ids.append(frame_name_to_id[f_name])
    kdtree_list = kdtree_list.astype('float32')
    return kdtree_list, frame_ids

