import argparse
import os
import glob
import torch.utils.data
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision.transforms as transforms
import torchvision


class CustomImageDataset(Dataset):
        def __init__(self, img_dir, transform=None):
            self.img_files = glob.glob(img_dir + '/**/*.jpg', recursive=True)
            self.transform = transform

        def __len__(self):
            return len(self.img_files)

        def __getitem__(self, idx):
            img_path = self.img_files[idx]
            image = Image.open(img_path)
            vid_class, vid_name, im_name = img_path.split('/')[-3], img_path.split('/')[-2], img_path.split('/')[-1]
            im_name = os.path.splitext(im_name)[0]
            if self.transform:
                image = self.transform(image)
            return image, vid_class, vid_name, im_name


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate VGG features."
    )
    parser.add_argument(
        '--data_path',
        type=str, required=True, help='Path to input data')
    parser.add_argument(
        '--out_path',
        default='output/features/vgg/', type=str, help='Path to output folder')
    parser.add_argument(
        "--batch_size",
        type=int, default=256,
        help="Batch size"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.vgg16(weights='VGG16_Weights.DEFAULT')
    model.classifier = model.classifier[:-2]
    model.to(device)
    model.eval()

    transform_RGB = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    train_dataloader = CustomImageDataset(args.data_path, transform=transform_RGB)
    loader = DataLoader(train_dataloader, batch_size=args.batch_size)

    cnt = 0
    total = len(loader)
    for i, (data, vid_classes, vid_names, im_names) in enumerate(loader):
        data = data.to(device)
        outputs = model(data)
        outputs = outputs.cpu().detach().numpy()
        for ix, output in enumerate(outputs):
            vid_class = vid_classes[ix]
            vid_name = vid_names[ix]
            im_name = im_names[ix]
            temp_path = f'{args.out_path}/{vid_class}/{vid_name}'
            os.makedirs(temp_path, exist_ok=True)
            np.save(f'{temp_path}/{im_name}.npy', output)
        cnt += 1
        print (f'Processed batch {cnt}/{total}')

