import argparse
import os
import numpy as np
import PIL
from PIL import Image, ImageFilter
from joblib import Parallel, delayed
import time
from tqdm import tqdm


def process_image(data_path, file, out_path, thumb_size, global_mean):
    
    start_time = time.time()

    vid_class, vid, frame = file
    f_name = os.path.splitext(frame)[0]
    save_path = os.path.join(out_path, vid_class, vid)
    os.makedirs(save_path, exist_ok=True)

    im = Image.open(os.path.join(data_path, vid_class, vid, frame))
    im = im.convert('L')
    im = im.resize((thumb_size, thumb_size), resample=PIL.Image.BICUBIC)
    # im.save(os.path.join(out_path, vid_class, vid, frame))
    im = np.array(im)
    im = im/255
    im = im - global_mean
    im = im.reshape(-1)
    np.save(os.path.join(save_path, f_name + '.npy'), im)

    elapsed_time = time.time() - start_time
    return elapsed_time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate thumbnails from original frames."
    )
    parser.add_argument(
        '--data_path',
        default="/storage4tb/PycharmProjects/GitHub/pytorch-center-loss_github/output/frames",
        type=str, required=False, help='Path to data')
    parser.add_argument(
        '--out_path',
        default="output/features/thumbnail/",
        type=str, help='Path to output data')
    parser.add_argument(
        '--thumb_size',
        default=12, type=int, help='Thumbnail size. Default: 12 -> 12x12')
    parser.add_argument(
        '--global_mean',
        default=0.386, type=float, help='Global data mean for normalization')
    parser.add_argument(
        "--cores",
        type=int, default=-1,
        help="Number of cores to use to process the data. Default: -1 -> Uses all cores."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    data_path = args.data_path
    out_path = args.out_path
    cores = os.cpu_count() if args.cores == -1 else args.cores
    thumb_size = args.thumb_size
    global_mean = args.global_mean

    dirs = os.listdir(data_path)
    all_files = []
    for vid_class in dirs:
        vids = os.listdir(os.path.join(data_path, vid_class))
        for vid in vids:
            frames = os.listdir(os.path.join(data_path, vid_class, vid))
            for frame in frames:
                all_files.append((vid_class, vid, frame))
    
    frame_times = Parallel(n_jobs=cores)(delayed(process_image)(data_path, file, out_path, thumb_size, global_mean) for file in tqdm(all_files, desc="Processing frames"))

    average_time = sum(frame_times) / len(frame_times)
    max_time = max(frame_times)
    min_time = min(frame_times)

    print(f"Processed {len(all_files)} frames")
    print(f"Average time per frame: {average_time:.4f} sec")
    print(f"Min time: {min_time:.4f} sec, Max time: {max_time:.4f} sec")

