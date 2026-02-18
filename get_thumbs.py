import os
import numpy as np
import PIL
from PIL import Image, ImageFilter
from joblib import Parallel, delayed
import time

def get_thumbs_func(data_path, vid_classes, out_path, thumb_size, kernel_size):
    i = 0
    total_time = 0
    cpt = sum([len(files) for r, d, files in os.walk(data_path)])
    
    # for vid_class in os.listdir(data_path):
    for vid_class in vid_classes:
        vids = os.listdir(os.path.join(data_path, vid_class))
        for vid in vids:
            frames = os.listdir(os.path.join(data_path, vid_class, vid))
            if not os.path.exists(os.path.join(out_path, vid_class, vid)):
                os.makedirs(os.path.join(out_path, vid_class, vid))
            for frame in frames:
                start_time = time.time()
                f_name = os.path.splitext(frame)[0]
                im = Image.open(os.path.join(data_path, vid_class, vid, frame))
                # im = im.filter(ImageFilter.GaussianBlur(radius=kernel_size))
                im = im.convert('L')

                im = im.resize((thumb_size, thumb_size), resample=PIL.Image.BICUBIC)
                im = np.array(im)
                im = im/255
                im = im - 0.386 # 0.329/vcsl 0.386/fivr
                im = im.reshape(-1)
                end_time = time.time()
                elapsed_time = end_time - start_time
                total_time += elapsed_time
                np.save(os.path.join(out_path, vid_class, vid, f_name + '.npy'), im)

                # im.save(os.path.join(out_path, vid_class, vid, frame + '.jpg'))
        i += 1
        print(i, '/', len(vid_classes))
    average_time = total_time / cpt
    print(f'Average time per iteration: {average_time:.4f} seconds')


if __name__ == "__main__":
    dirs = os.listdir(data_path)
    two_split = np.array_split(dirs, num_of_cores)
    pths = []
    for array in two_split:
        pths.append(list(array))
    Parallel(n_jobs=num_of_cores, prefer="threads")(delayed(get_thumbs_func)(data_path, x, out_path, thumb_size, kernel_size) for x in pths)


    '''uncomment to get single core time consumption to get thumbnails'''
    # data_path = '/storage4tb/PycharmProjects/pytorch-center-loss/vcsl_data_process/output/vcsl_frames_test/'
    # out_path = '/storage4tb/PycharmProjects/pytorch-center-loss/vcsl_data_process/output/temp/'
    # num_of_cores = 20
    # thumb_size = 12
    # kernel_size = 1.2
    # dirs = os.listdir(data_path)
    # get_thumbs_func(data_path, dirs, out_path, thumb_size, kernel_size)

    
