import argparse
import os
import numpy as np
import h5py
import scipy
import natsort
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make a single .h5 file for features")

    parser.add_argument("--feature_path", type=str, required=True,
                        help="Path to features (.npy or .mat)")
    parser.add_argument('--save_file', type=str, required=True,
                        help='Path to output h5 file (including output filename and extension)')

    args = parser.parse_args()

    feature_path = args.feature_path
    save_file = args.save_file

    cpt = sum([len(files) for _, _, files in os.walk(feature_path)])

    cnt = 0
    with h5py.File(save_file, 'w', libver='latest') as hf:
        with tqdm(total=cpt, desc="Processing features") as pbar:
            for vid_class in os.listdir(feature_path):
                for vid in os.listdir(os.path.join(feature_path, vid_class)):
                    for frame in natsort.natsorted(os.listdir(os.path.join(feature_path, vid_class, vid))):
                        if frame.endswith('.npy'):
                            feature = np.load(os.path.join(feature_path, vid_class, vid, frame))
                            feature = feature.reshape(1, -1)[0]
                            f_name = frame.replace('.npy', '')
                        elif frame.endswith('.mat'):
                            feature = scipy.io.loadmat(os.path.join(feature_path, vid_class, vid, frame))
                            feature = feature['data'][0]
                            f_name = frame.replace('-f_scfv.mat', '')
                            f_name = f_name.replace('.mat', '')
                        ts = int(f_name.split('-')[-1])
                        frame_id = cnt
                        data = feature
                        
                        dataset_name = str(frame_id)
                        hf.create_dataset(dataset_name, data=data)

                        cnt += 1
                        pbar.update(1)
                
