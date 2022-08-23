import torch
from torch.utils.data import Dataset

from skimage import io, transform
from skimage.color import rgb2lab

import os
import numpy as np

size_im = 256

def tones_mask(image2):
    threshold = 255 / 3
    dark_tones_mask = image2 < threshold
    mid_tones_mask = (threshold < image2) & (image2 < 2 * threshold)
    light_tones_mask = image2 > 2 * threshold
    return dark_tones_mask, mid_tones_mask, light_tones_mask

class RGB_tones_mask(object):
    def __init__(self):
        self.Tones_transform = tones_mask()

    def __call__(self, image, id):
        bw = rgb2lab(image)[:, :, 0] # get the bw version
        Y = image[:, :, id] # selects the color chanel

        dark_tones = Y * self.Tones_transform[0] #dark_tones_mask
        light_tones = Y * self.Tones_transform[1] #light_tones_mask
        mid_tones = Y * self.Tones_transform[2] #mid_tones_mask

        bw_mask = bw * dark_tones
        image_mask = image*dark_tones


        #sample = {'image_bw': X, 'image_ab': Y}
        #image_bw, image_ab = sample['image_bw'], sample['image_ab']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image_bwT = torch.from_numpy(bw_mask).float()
        image_bwT = image_bwT.unsqueeze(dim=0)  # torch.Size([1, 400, 200]) it is like transpose((2, 0, 1))

        image_maskT = Y.transpose((2, 0, 1))
        #image_abT = image_ab.transpose((2, 1, 0))
        image_maskT = torch.from_numpy(image_maskT).float()
        #image_abT = torch.from_numpy(image_ab).float()
        #image_abT = image_abT.unsqueeze(dim=0)

        return {'bw_mask': image_bwT, 'image_mask': image_maskT}

class ImgDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.lst_names = os.listdir(root_dir)
        self.tones_transform = RGB_tones_mask()
        self.transform = transform # for the augmentation image_X

    def __len__(self):
        return len(self.lst_names) # of how many examples(images?) you have

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.lst_names[idx])
        image = io.imread(img_name) # numpy array

        sample = transform.resize(image, (size_im, size_im))

        if self.transform:
            sample = self.transform(sample)

        if self.tones_transform:
            sample = self.tones_transform(sample)

        # check an image or dict with two keys

        return sample


