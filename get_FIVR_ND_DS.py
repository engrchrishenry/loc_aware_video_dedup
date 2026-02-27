import json

ann_path = 'dataset/annotation.json'

f = open(ann_path)
data = json.load(f)
s_f = open('seed_vid_ids.txt', 'w')
nd_ds_f = open('youtube_ids_ND_DS.txt', 'w')
id = 1
id_to_query_id = {}
for seed_vid in data:
    s_f.write(seed_vid + ' ' + str(id) + '\n')
    id_to_query_id[seed_vid] = id
    ds_names = data[seed_vid]["DS"]
    nd_ds_f.write(seed_vid + '\n')
    for ds_name in ds_names:
        id_to_query_id[ds_name] = str(id)
        nd_ds_f.write(ds_name + '\n')
    if "ND" in data[seed_vid].keys():
        nd_names = data[seed_vid]["ND"]
        for nd_name in nd_names:
            id_to_query_id[nd_name] = str(id)
            nd_ds_f.write(nd_name + '\n')
    id += 1
json.dump(id_to_query_id, open("id_to_query_id.json", "w"))
s_f.close()
nd_ds_f.close()
