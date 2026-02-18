import numpy as np


def get_nodes_scipy(tree_thumb, tree_fv, tree_vgg, thumb_test, fv_test, vgg_test, k):
    _, pred_nodes_thumb = tree_thumb.query(thumb_test, k=k, workers=-1)
    _, pred_nodes_fv = tree_fv.query(fv_test, k=k, workers=-1)
    _, pred_nodes_vgg = tree_vgg.query(vgg_test, k=k, workers=-1)
    nodes = []
    for i in range(len(pred_nodes_thumb)):
        nodes.append(np.unique(np.concatenate((pred_nodes_thumb[i], pred_nodes_fv[i], pred_nodes_vgg[i]),0)).tolist())
    return nodes


def prune_vid_ids(nodes, indx_info_dict):
    '''Prune video IDs'''
    
    temp_nodes = nodes[1:]
    for i in range(len(nodes)):
        if i >= len(nodes) - 1:
            break
        unique_vid_ids = np.unique([indx_info_dict[y][1] for y in nodes[i]])
        temp_node = nodes[i+1]
        temp = []
        for j in range(len(temp_node)):
            vid_id = indx_info_dict[temp_node[j]][1]
            if vid_id in unique_vid_ids:
                temp.append(temp_node[j])
        nodes[i + 1] = temp

    unique_vid_ids = [indx_info_dict[x][1] for y in temp_nodes for x in y]
    unique_vid_ids = np.unique(unique_vid_ids)
    temp_node = nodes[0]
    temp = []
    for frame_id in temp_node:
        vid_id = indx_info_dict[frame_id][1]
        if vid_id in unique_vid_ids:
            temp.append(frame_id)
    nodes[0] = temp

    return nodes


def prune_ts(nodes, indx_info_dict, ts, tolerance):
    '''Prune timestamps'''

    for i in range(len(nodes)):
        if i >= len(nodes) - 1:
            break
        current_node = nodes[i]
        next_node = nodes[i + 1]
        temp = []
        for j in range(len(next_node)):
            nex_time = indx_info_dict[next_node[j]][3]
            indicator = 0
            for k in range(len(current_node)):
                cur_time = indx_info_dict[current_node[k]][3]
                if ts - tolerance <= (nex_time - cur_time) <= ts + tolerance:
                    indicator = 1
                    temp.append(next_node[j])
                    break


        nodes[i + 1] = temp
        
    temp = []
    for s1_element in nodes[0]:
        s1_time = indx_info_dict[s1_element][3]
        indicator = 0
        for s2_element in nodes[1]:
            s2_time = indx_info_dict[s2_element][3]
            if ts - tolerance <= (s2_time - s1_time) <= ts + tolerance:
                indicator = 1
                temp.append(s1_element)
                break
    nodes[0] = temp
    return nodes


def retrieve_vid_ids(final_nodes_vid_id):
    retrieved_vid_ids = [list(set(lst)) for lst in final_nodes_vid_id]
    retrieved_vid_ids = [l for l in retrieved_vid_ids if l]
    print('number of frames retrieved after taking common:', [len(x) for x in retrieved_vid_ids])
    if retrieved_vid_ids != []:
        retrieved_vid_ids = set(retrieved_vid_ids[0]).intersection(*retrieved_vid_ids[1:])
    else:
        retrieved_vid_ids = set()
    return retrieved_vid_ids


def retrieve_frame_ids(final_nodes_frame_id, num_of_query_frames, ts, num_of_frames):
    retrieved_frame_ids = []
    prev_sets = final_nodes_frame_id[:-1]
    last_set = final_nodes_frame_id[-1]
    for frame_id in last_set:
        indicator_temp = 0
        vid_id_temp = ''.join([xc + '-' for xc in frame_id.split('-')[:-1]])[:-1]
        end_time_temp = int(frame_id.split('-')[-1])
        float_values = np.arange((end_time_temp * num_of_frames) - (ts * (num_of_query_frames - 1)), (end_time_temp * num_of_frames), ts)/num_of_frames
        int_values = float_values.astype(int)
        for p, set_temp in enumerate(prev_sets):
            n_temp = frame_id.replace(f'-{end_time_temp}', f'-{int_values[p]}')
            if n_temp in set_temp:
                indicator_temp += 1
        if indicator_temp == num_of_query_frames - 1:
            retrieved_frame_ids.append(f'{vid_id_temp}-{int_values[0]}-{end_time_temp}')
    return retrieved_frame_ids


def cal_acc_ts(gt_vid_id, gt_start_time, gt_end_time, retrieved_frame_ids, acc_ts_th, acc_vid_id_ts, num_of_frames):
    retrieved_frame_ids = [s for s in retrieved_frame_ids if
                         gt_vid_id in s]
    if retrieved_frame_ids != []:
        retrieved_ts = [list(map(float, k.split('-')[-2:])) for k in retrieved_frame_ids]
        temp1 = []
        for temp_ts in retrieved_ts:
            min_start_pred_time = min(temp_ts)*num_of_frames
            max_start_pred_time = max(temp_ts)*num_of_frames
            temp2 = []
            for b in acc_ts_th:
                if ((gt_start_time - b) <= min_start_pred_time <= (gt_start_time + b)) and (
                    (gt_end_time - b) <= max_start_pred_time <= (gt_end_time + b)):
                    temp2.append(1)
                else:
                    temp2.append(0)
            temp1.append(temp2)
        print (temp1)
        temp1 = [int(any(item)) for item in zip(*temp1)]
        print (temp1)
        acc_vid_id_ts.append(temp1)
    return acc_vid_id_ts

