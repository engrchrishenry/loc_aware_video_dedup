import argparse
import os
import natsort
import pickle
import gzip
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate index file")

    parser.add_argument('--frames_path', type=str, required=True,
        help='Path to frames')
    parser.add_argument('--save_file', type=str, required=True,
        help='Path to output .pickle file (including output filename and extension)')
    parser.add_argument('--frame_interval', type=float, default=0.5,
        help='Frame sampling interval in seconds.'
            'Default: 0.5 (extracts 1 frame every 0.5 seconds). '
            'Example: 30 -> extracts 1 frame every 30 seconds.')

    args = parser.parse_args()

    frames_path = args.frames_path
    save_file = args.save_file
    frame_interval = args.frame_interval

    cpt = sum([len(files) for _, _, files in os.walk(frames_path)])

    my_dict = {}
    cnt = 0
    with tqdm(total=cpt, desc="Generating indexes") as pbar:
        for vid_class in os.listdir(frames_path):
            for vid_name in os.listdir(os.path.join(frames_path, vid_class)):
                for frame in natsort.natsorted(os.listdir(os.path.join(frames_path, vid_class, vid_name))):
                    frame_name, ext = os.path.splitext(frame)
                    my_dict[cnt] = [vid_class, vid_name, frame_name, round(cnt*frame_interval, 1)]
                    cnt += 1
                    pbar.update(1)

    print ('Saving indexes to .pickle file')
    with gzip.open(save_file, 'wb') as handle:
        pickle.dump(my_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

