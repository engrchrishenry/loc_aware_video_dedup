import argparse
import numpy as np
import pickle
import prune_utils
import time
import pandas as pd
import gzip
import yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast Video Deduplication and Localization With Temporal Consistence Re-Ranking")

    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file")
    
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    repo_kdtrees = config["repo_kdtrees"]
    repo_path_thumb_test = config["repo_thumb_test"]
    repo_path_fv_test = config["repo_fv_test"]
    repo_path_vgg_test = config["repo_vgg_test"]
    repo_indexes = config["repo_indexes"]
    repo_indexes_test = config["repo_indexes_test"]
    out_path = config["out_path"]

    ts = config["ts"]
    tolerance = config["tolerance"]
    k_values = config["k_values"]
    acc_ts_th = config["acc_ts_th"]
    frame_interval = config["frame_interval"]

    indx_info_dict = pickle.load(gzip.open(repo_indexes, 'rb'))
    name_to_frame_id = {}
    name_to_vid_id = {}
    for frame_id, details in indx_info_dict.items():
        name_to_frame_id[details[2]] = frame_id
    indx_info_test_dict = pickle.load(gzip.open(repo_indexes_test, 'rb'))
    gt_vid_ids = [details[1] for _, details in indx_info_test_dict.items()]
    gt_frame_names = [details[2] for _, details in indx_info_test_dict.items()]

    frame_ids = list(range(0, len(indx_info_dict), 1))

    print('--------------------------------------Loading tree--------------------------------------')
    tree_data = pickle.load(open(repo_kdtrees, "rb"))
    tree_thumb = tree_data['tree_thumb']
    tree_fv = tree_data['tree_fv']
    tree_vgg = tree_data['tree_vgg']

    print('--------------------------------------Loading test repo--------------------------------------')
    repo_thumb_test = np.load(repo_path_thumb_test).squeeze(axis=1)
    repo_fv_test = np.load(repo_path_fv_test).squeeze(axis=1)
    repo_vgg_test = np.load(repo_path_vgg_test).squeeze(axis=1)
    print (f'Thumbnail test repo shape = {repo_thumb_test.shape}')
    print (f'Fisher vector test repo shape = {repo_fv_test.shape}')
    print (f'VGG test repo shape = {repo_vgg_test.shape}')

    result_all_f = open(f'{out_path}/res_all.txt', 'w')
    for k in k_values:    
        result_k_f = open(f'{out_path}/res_k_{k}.txt', 'w')
        recall = 0
        # recall_neg = 0
        t_total_query = []
        t_node_query = []
        t_prune_query = []
        acc_vid_id_ts = []
        i = 1
        for gt_vid_id in np.unique(gt_vid_ids):
            print ('------------------------------', k, i, '/', len(np.unique(gt_vid_ids)), '------------------------------')
            i += 1

            indxs = np.where(np.array(gt_vid_ids) == gt_vid_id)[0]
            num_of_query_frames = len(indxs)
            gt_start_num = int(gt_frame_names[indxs[0]].split('-')[-1])
            gt_end_num = int(gt_frame_names[indxs[-1]].split('-')[-1])
            gt_start_time = int(gt_frame_names[indxs[0]].split('-')[-1])*frame_interval
            gt_end_time = int(gt_frame_names[indxs[-1]].split('-')[-1])*frame_interval
            
            start_time = time.time()
            nodes = prune_utils.get_nodes_scipy(tree_thumb, tree_fv, tree_vgg, repo_thumb_test[indxs], repo_fv_test[indxs], repo_vgg_test[indxs], k=k)
            nodes_time = time.time()-start_time
            t_node_query.append(nodes_time)

            print('# of frames retrieved before vid_ids pruning:', [len(x) for x in nodes])
            temp = time.time()
            nodes = prune_utils.prune_vid_ids(nodes, indx_info_dict)
            prune_vid_id_time = time.time()-temp
            print('number of frames retrieved after vid_ids pruning:', [len(x) for x in nodes])
            temp = time.time()
            nodes = prune_utils.prune_ts(nodes, indx_info_dict, ts, tolerance)
            prune_ts_time = time.time()-temp
            t_prune_query.append(prune_vid_id_time + prune_ts_time)
            print('number of frames retrieved after timestamps pruning:', [len(x) for x in nodes])

            final_nodes_vid_id = [[indx_info_dict[idx][1] for idx in node] for node in nodes]
            final_nodes_frame_id = [[indx_info_dict[idx][2] for idx in node] for node in nodes]

            retrieved_vid_ids = prune_utils.retrieve_vid_ids(final_nodes_vid_id)
            print(f'gt vid id is {gt_vid_id}, Final retrieval {retrieved_vid_ids}')
            
            if retrieved_vid_ids != set():
                retrieved_frame_ids = prune_utils.retrieve_frame_ids(final_nodes_frame_id, num_of_query_frames, ts, frame_interval)
                print(f'gt vid id is {gt_vid_id}, \
                gt ts={gt_start_time}-{gt_end_time}, \
                gt num={gt_start_num}-{gt_end_num} \
                Final retrieval {retrieved_frame_ids}')

            final_time = time.time()-start_time
            t_total_query.append(final_time)

            final_nodes_frame_idx = pd.DataFrame(final_nodes_frame_id, dtype=object)
            pd.set_option('display.max_colwidth', None)
            result_k_f.writelines(f'\n------------------------------{gt_vid_id}/{gt_start_time}-{gt_end_time}-----------------------------\n')
            if retrieved_vid_ids != set():
                result_k_f.writelines(f'------------------------------{retrieved_frame_ids}-----------------------------\n')
            result_k_f.writelines(final_nodes_frame_idx.T.to_string())
            gt_vid_id_ts = f'{gt_vid_id}-{gt_start_time}-{gt_end_time}'

            if retrieved_vid_ids != set():
                acc_vid_id_ts = prune_utils.cal_acc_ts(gt_vid_id, float(gt_start_time), float(gt_end_time), retrieved_frame_ids, acc_ts_th, acc_vid_id_ts, frame_interval)

            if gt_vid_id in retrieved_vid_ids:
                recall += 1
            # if retrieved_vid_ids == set() or retrieved_frame_ids == []:
            #     recall_neg  += 1

            print (f'Nodes search time = {(nodes_time)*1000:.4f} mseconds')
            print (f'Prune vid_id time = {(prune_vid_id_time)*1000:.4f} mseconds')
            print (f'Prune ts time = {(prune_ts_time)*1000:.4f} mseconds')
            print (f'Final search time = {(final_time)*1000:.4f} mseconds')

        result_k_f.close()

        avg_t_node_frame = np.mean(t_node_query)/len(indxs)*1000
        avg_t_node_query = np.mean(t_node_query)*1000
        avg_t_prune_frame = np.mean(t_prune_query)/len(indxs)*1000
        avg_t_prune_query = np.mean(t_prune_query)*1000
        avg_t_total_frame = np.mean(t_total_query)/len(indxs)*1000
        avg_t_total_query = np.mean(t_total_query)*1000
        recall = recall/len(np.unique(gt_vid_ids))*100
        # recall_neg = recall_neg/len(np.unique(gt_vid_ids))*100
        acc_vid_id_ts = np.array(acc_vid_id_ts)
        acc_vid_id_ts = np.sum(acc_vid_id_ts, 0)/len(np.unique(gt_vid_ids))*100

        acc_str = "\n".join(
            f"  threshold={t:<5} → {a:.4f}%"
            for t, a in zip(acc_ts_th, acc_vid_id_ts)
        )

        res_print = (f'k = {k}, ts {ts}, tolerance {tolerance}\n'
            f'Avg node search time/frame = {avg_t_node_frame:.4f} ms\n'
            f'Avg node search time/query = {avg_t_node_query:.4f} ms\n'
            f'Avg prune time/frame = {avg_t_prune_frame:.4f} ms\n'
            f'Avg prune time/query = {avg_t_prune_query:.4f} ms\n'
            f'Avg total time/frame = {avg_t_total_frame:.4f} ms\n'
            f'Avg total time/query = {avg_t_total_query:.4f} ms\n'
            f'Recall = {recall}\n'
            # f'Timestamp accuracy at varying tolerance thresholds = {acc_ts_th}{acc_vid_id_ts}\n'
            f'Timestamp accuracy at varying tolerance levels:\n{acc_str}\n'
            # f'Recall negative = {recall_neg}\n'
            f'------------------------------------------\n'
        )

        result_all_f.writelines(res_print)
        
        print (res_print)
    
    result_all_f.close()
    