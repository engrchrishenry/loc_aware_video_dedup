import argparse
import cv2
from PIL import Image
import PIL
import random
from skimage.exposure import rescale_intensity
from skimage.segmentation import slic
from skimage.util import img_as_float
from sklearn.cluster import MiniBatchKMeans
from skimage import io
import numpy as np
import math
import os
from math import log10, sqrt
from natsort import natsorted
import matplotlib.pyplot as plt


def adjust_gamma(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def hue_image(image, hue):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = image[:, :, 0]
    v = np.where(v + hue <= 255, v + hue, 255)
    image[:, :, 0] = v
    image2 = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image2


def gausian_blur(image, blur):
    image2 = cv2.GaussianBlur(image, (5, 5), blur)
    return image2


def quantization(image, ratio):
    crop_height = int(image.shape[0] * math.sqrt(ratio))
    crop_width = int(image.shape[1] * math.sqrt(ratio))
    nh = random.randint(0, image.shape[0] - crop_height)
    nw = random.randint(0, image.shape[1] - crop_width)
    image_crop = image[nh:nh + crop_height, nw:nw + crop_width]
    return image_crop


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate test data (normal version)")

    parser.add_argument("--test_vid_list", type=str, required=True,
                        help="Path to test video list")
    parser.add_argument("--frames_path", type=str, required=True,
                        help="Path to frames")
    parser.add_argument("--out_path", type=str, required=True,
                        help="Path to output data")
    parser.add_argument("--frame_interval", type=float, required=True,
                        help="Frame interval value used while extracting frames with extract_frames.py")
    parser.add_argument("--num_of_sec", type=float, default=40.0,
                        help="Number of seconds to extract from video")
    parser.add_argument("--ts", type=float, default=10.0,
                        help="Time step for generating test query")
    parser.add_argument("--psnr_thresh", type=float, default=10.0,
                        help="PSNR threshold. Maintain atleast this PSNR after augmentations.")
    parser.add_argument("--keypoint_thresh", type=int, default=50,
                        help="SIFT keypoint threshold")
    
    args = parser.parse_args()

    cnt = 0
    psnr_values = []
    vid_path_file = open(args.test_vid_list, 'r')
    vid_paths = vid_path_file.readlines()
    sift = cv2.SIFT_create()
    for vid_ann in vid_paths:
        vid_ann = vid_ann.replace('\n', '')
        vid_class, vid_fname = vid_ann.split(' ')[0].split('/')
        print ('Processing', vid_class, '/', vid_fname)
        
        vid_frames = natsorted(vid_ann.split(' ')[1:])

        aug_list = ['hue_image', 'gausian_blur', 'adjust_gamma', 'quantization']

        if not os.path.exists(os.path.join(args.out_path, vid_class, vid_fname)):
            os.makedirs(os.path.join(args.out_path, vid_class, vid_fname))
        # if not os.path.exists(os.path.join(args.out_path, 'plots', vid_class, vid_fname)):
        #     os.makedirs(os.path.join(args.out_path, 'plots', vid_class, vid_fname))
        
        random.shuffle(aug_list)
        aug_list = aug_list[:3]
        hue_temp = random.randint(0, 20)
        gausian_temp = random.uniform(0.0, 5.0)
        ratio = random.uniform(0.5, 0.9)
        gamma = 0.5
        temp = random.randint(30, 90)
        min_time = float(vid_frames[0].split('-')[-1].split('.')[0]) * args.frame_interval
        max_time = float(vid_frames[-1].split('-')[-1].split('.')[0]) * args.frame_interval
        
        for vid_frame in vid_frames:
            psnr_th = 0
            num_kps = 0
            while psnr_th < args.psnr_thresh and num_kps < args.keypoint_thresh:
                image = cv2.imread(os.path.join(args.frames_path, vid_class, vid_fname, vid_frame))
                ori_image = cv2.imread(os.path.join(args.frames_path, vid_class, vid_fname, vid_frame))
                for aug in aug_list:
                    if aug == 'hue_image':
                        image = hue_image(image, hue_temp)
                    elif aug == 'gausian_blur':
                        image = gausian_blur(image,gausian_temp)
                    elif aug == 'adjust_gamma':
                        image = adjust_gamma(image, ratio)
                if 'quantization' in aug_list:
                    cv2.imwrite('temp.jpg', image)
                    im = Image.open('temp.jpg')
                    im.save('temp.jpg', quality=temp, optimize=True, progressive=True)
                    image = cv2.imread('temp.jpg')
                psnr_th = PSNR(ori_image, image)
                keypoints, descriptors = sift.detectAndCompute(image, None)
                num_kps = len(keypoints)
            psnr_values.append(psnr_th)
            cv2.imwrite(os.path.join(args.out_path, vid_class, vid_fname, str(vid_frame) + '.jpg'), image)
            
            fig, axes = plt.subplots(1, 2)
            axes[0].imshow(ori_image)
            axes[0].axis('off')
            axes[1].imshow(image)
            axes[1].axis('off')
            plt.tight_layout()
            # plt.savefig(os.path.join(args.out_path, 'plots', vid_class, vid_fname, str(vid_frame) + '.jpg'))
            plt.close()

        cnt += 1
        print('Processed', str(cnt), '/', len(vid_paths), '\n')

    plt.hist(psnr_values)
    plt.xlim(20, 50)
    # plt.savefig('hist.jpg')
    plt.close()
