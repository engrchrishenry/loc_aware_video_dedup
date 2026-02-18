import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
from sklearn.decomposition import PCA, IncrementalPCA
import matplotlib.pyplot as plt
import pickle
import os
from PIL import Image
import time

data_path = '/storage4tb/PycharmProjects/pytorch-center-loss/vcsl_data_process/output/vcsl_frames_test/'
out_path = '/storage4tb/PycharmProjects/pytorch-center-loss/vcsl_data_process/output/temp/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
# model = torchvision.models.vgg16(weights='VGG16_Weights.DEFAULT')
model.classifier = model.classifier[:-2]
model.eval()

print('-----------------------Computing feature vectors.-----------------------')
i = 0
total_time = 0
cpt = sum([len(files) for r, d, files in os.walk(data_path)])
for vid_class in os.listdir(data_path):
    vids = os.listdir(os.path.join(data_path, vid_class))
    v = 1
    for vid in vids:
        frames = os.listdir(os.path.join(data_path, vid_class, vid))
        if not os.path.exists(os.path.join(out_path, vid_class, vid)):
            os.makedirs(os.path.join(out_path, vid_class, vid))
        for frame in frames:
            start_time = time.time()
            im = Image.open(os.path.join(data_path, vid_class, vid, frame))
            im = im.resize((224, 224))
            t = transforms.ToTensor()
            im = t(im).to(device)
            model.to(device)
            with torch.no_grad():
                output = model(im.unsqueeze(0))
                output = output.cpu().numpy()
                end_time = time.time()
                elapsed_time = end_time - start_time
                total_time += elapsed_time
                np.save(os.path.join(out_path, vid_class, vid, frame + '.npy'), output)
        print(v, '/', len(vids))
        v += 1
    i += 1
    print(i, '/', len(os.listdir(data_path)))
print(f'Average time per iteration: {total_time / cpt:.4f} seconds')
