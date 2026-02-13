import argparse
import os
import natsort
import numpy as np
import scipy.io
import gc

def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate global mean for thumbnail feature normalization."
    )
    parser.add_argument(
        '--feature_path',
        type=str, required=True, help='Path to features (.npy or .mat)')
    parser.add_argument(
        '--save_file',
        type=str, help='Path to output file (including ouput filename and extension)')
    parser.add_argument(
        "--frame_interval",
        type=float,
        help="Frame sampling interval in seconds."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cpt = sum([len(files) for _, _, files in os.walk(args.feature_path)])

    kdtree_list = []
    cnt = 0
    kdtree_list = np.zeros(shape=(cpt, 6), dtype=object)
    ts_c = args.frame_interval
    for vid_class in os.listdir(args.feature_path):
        for vid in os.listdir(os.path.join(args.feature_path, vid_class)):
            for frame in natsort.natsorted(os.listdir(os.path.join(args.feature_path, vid_class, vid))):
                if frame.endswith('.npy'):
                    feature = np.load(os.path.join(args.feature_path, vid_class, vid, frame))
                    feature = feature.reshape(1, -1)
                    f_name = frame.replace('.npy', '')
                elif frame.endswith('.mat'):
                    feature = scipy.io.loadmat(os.path.join(args.feature_path, vid_class, vid, frame))
                    feature = feature['data']
                    f_name = frame.replace('-f_scfv.mat', '')
                    f_name = f_name.replace('.mat', '')
                frame_id = cnt
                kdtree_list[cnt, 0] = frame_id
                kdtree_list[cnt, 1] = f_name
                kdtree_list[cnt, 2] = feature
                kdtree_list[cnt, 3] = vid_class
                kdtree_list[cnt, 4] = vid
                kdtree_list[cnt, 5] = ts_c
                cnt += 1
                ts_c += args.frame_interval
                gc.collect()
                if cnt % 10000 == 0:
                    print (f'Processed feature {cnt}/{cpt}')

    kdtree_list = np.array(kdtree_list)
    print(kdtree_list.shape)
    np.save(args.save_file, kdtree_list)
