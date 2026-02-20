import argparse
import glob
import os
import random
import natsort


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select test videos from dataset frames")

    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to frames extracted via extract_frames.py")
    parser.add_argument("--num_of_vids", default=1000, type=int,
                        help="Number of videos to select")
    parser.add_argument("--vid_per_class", default=10, type=int,
                        help="Number of videos to select per class")
    parser.add_argument("--min_duration", default=41.0, type=float,
                        help="Minimum duration of video in seconds")
    parser.add_argument("--frame_interval", default=0.5, type=float,
                        help="Frame interval. Choose the same the one while using extract_frames.py")
    parser.add_argument('--save_file', type=str, required=True,
                        help='Path to output text file (including ouput filename and extension)')

    
    args = parser.parse_args()
    
    vid_files_final = []
    for vid_class in os.listdir(args.data_path):
        count = 0
        for vid in os.listdir(os.path.join(args.data_path, vid_class)):
            vid_frames = natsort.natsorted(os.listdir(os.path.join(args.data_path, vid_class, vid)))
            max_time = round(len(vid_frames)*args.frame_interval, 1)
            if args.min_duration <= max_time:
                vid_files_final.append(vid_class + '/' + vid)
                count += 1
            if count == args.vid_per_class:
                break

    vid_files_final_new = []
    if len(vid_files_final) != args.num_of_vids:
        remaining_num_of_vids = args.num_of_vids - len(vid_files_final)
        vid_files = glob.glob(args.data_path + '/*/*')
        random.shuffle(vid_files)
        # print (len(vid_files))
        for vid in vid_files:
            if vid.split('/')[-2] + '/' + vid.split('/')[-1] not in vid_files_final:
                vid_frames = natsort.natsorted(os.listdir(vid))
                max_time = round(len(vid_frames)*args.frame_interval, 1)
                if args.min_duration <= max_time:
                    vid_files_final_new.append(vid.split('/')[-2] + '/' + vid.split('/')[-1])
                if len(vid_files_final_new) == remaining_num_of_vids:
                    break

    common_elements = set(vid_files_final).intersection(vid_files_final_new)
    # print(common_elements, common_elements)

    vid_files_final = vid_files_final + vid_files_final_new

    file = open(args.save_file, 'w')
    for items in vid_files_final:
        file.writelines(items + '\n')
    file.close()

    print ('Completed.')

    