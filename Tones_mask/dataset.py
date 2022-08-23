import torch
from torch.utils.data import Dataset

from skimage import io, transform
from skimage.color import rgb2gray


import matplotlib.pyplot as plt

import os
import numpy as np

size_im = 400 #256

def tones_mask(image2, id):
    threshold = 1.0 / 3
    if id == 0:  return(image2 < threshold) # dark_tones_mask
    if id == 1:  return((threshold < image2) & (image2 < 2 * threshold)) # mid_tones_mask
    if id == 2:  return(image2 > 2 * threshold) #light_tones_mask

class RGB_tones_mask(object):
    def __call__(self, image, id):
        image = transform.resize(image, (size_im, size_im))
        #fig = plt.figure()
        #plt.imshow(image)
        #fig.savefig('image_orig_reco.png')

        bw = rgb2gray(image)
        Y = image[:, :, id] # selects the color chanel

        #dark_tones = image * tones_mask(image,id) #dark_tones_mask
        #light_tones = image * tones_mask(image)[1] #light_tones_mask
        #mid_tones = image * tones_mask(image)[2] #mid_tones_mask

        bw_mask = bw * tones_mask(bw,id)
        image_mask = image*tones_mask(image,id)


        #sample = {'image_bw': X, 'image_ab': Y}
        #image_bw, image_ab = sample['image_bw'], sample['image_ab']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image_bwT = torch.from_numpy(bw_mask).float()
        image_bwT = image_bwT.unsqueeze(dim=0)  # torch.Size([1, 400, 200]) it is like transpose((2, 0, 1))

        image_maskT = image_mask.transpose((2, 0, 1))
        #image_abT = image_ab.transpose((2, 1, 0))
        image_maskT = torch.from_numpy(image_maskT).float()

        return {'bw_mask': image_bwT, 'image_mask': image_maskT}

class ImgDataset(Dataset):
    def __init__(self, root_dir, id_tone):
        self.root_dir = root_dir
        self.lst_names = os.listdir(root_dir)
        self.tones_transform = RGB_tones_mask()
        self.id_tone = id_tone

    def __len__(self):
        return len(self.lst_names) # of how many examples(images?) you have

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.lst_names[idx])
        image = io.imread(img_name) # numpy array

        #sample = transform.resize(image, (size_im, size_im))

        if self.tones_transform:
            sample = self.tones_transform(image, self.id_tone)

        # check an image or dict with two keys

        return sample


def buildImagefromMask(X, Y, output):
    X = X.detach().numpy()
    X = X[0].transpose((1, 2, 0))
    X = np.squeeze(X, axis=2)

    Y = Y.detach().numpy()
    Y = Y[0].transpose((1, 2, 0))


    output = output.detach().numpy()  # output.numpy().transpose((1, 2, 0))
    output = output[0].transpose((1, 2, 0))

    return X, Y, output