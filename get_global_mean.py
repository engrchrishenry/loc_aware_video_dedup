import argparse
import os
from PIL import Image
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate global mean for thumbnail feature normalization."
    )
    parser.add_argument(
        '--data_path',
        type=str, required=True, help='Path to data')
    parser.add_argument(
        '--thumb_size',
        default=12, type=int, help='Thumbnail size. Default: 12 -> 12x12')
    parser.add_argument(
        "--cores",
        type=int, default=-1,
        help="Number of cores to use to process the data. Default: -1 -> Uses all cores."
    )

    return parser.parse_args()


def get_image_mean(file, thumb_size):
    im = Image.open(file).convert('L')
    im = im.resize((thumb_size, thumb_size))
    im = np.array(im)
    im = im/255
    sum_im = np.sum(im)
    n_pixs = thumb_size * thumb_size

    return sum_im, n_pixs


if __name__ == "__main__":
    args = parse_args()

    all_files = [f"{root}/{f}" for root, dirs, files in os.walk(args.data_path) for f in files]

    results = Parallel(n_jobs=args.cores)(delayed(get_image_mean)(file, args.thumb_size) for file in tqdm(all_files, desc="Processing frames"))

    sum_list, npix_list = zip(*results)

    global_average_mean = sum(sum_list) / sum(npix_list)
    print (global_average_mean)



