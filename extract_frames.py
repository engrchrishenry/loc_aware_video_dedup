import argparse
import os
import subprocess
from multiprocessing import Pool
import glob

def process_video(file):
    
    
    f_name, f_ext = os.path.splitext(os.path.basename(file))
    vid_class = file.split('/')[-2]
    
    out_path_temp = os.path.join(out_path, vid_class, f_name)
    if not os.path.exists(out_path_temp):
        os.makedirs(out_path_temp)

    query = "ffmpeg -i " + os.path.join(data_path, file) + " -vf fps=1/" + str(frame_interval) + " " + out_path_temp + f"/{f_name}-%d.jpg"
    subprocess.run(query, shell=True, stdout=subprocess.PIPE)

    print('Processed video:', file)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract frames for video file using ffmpeg and multi-core processing."
    )
    parser.add_argument(
        '--data_path',
        required=True,
        type=str, help='Path to input data')
    parser.add_argument(
        '--out_path',
        default="output/frames/",
        type=str, help='Path to output data')
    parser.add_argument(
        '--frame_interval',
        type=float,
        default=0.5,
        help='Frame sampling interval in seconds.'
            'Default: 0.5 (extracts 1 frame every 0.5 seconds). '
            'Example: 30 -> extracts 1 frame every 30 seconds.')
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
    frame_interval = args.frame_interval
    cores = os.cpu_count() if args.cores == -1 else args.cores
    
    # video_files = os.listdir(data_path)
    video_files = glob.glob(os.path.join(data_path, '**', '*'))

    # Create a pool of worker processes
    with Pool(processes=cores) as pool:
        # Map the process_video function to each video file in parallel
        pool.map(process_video, video_files)
