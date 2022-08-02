import torch
from torch.utils.data import Dataset

from skimage import io, transform
from skimage.color import rgb2lab, lab2rgb

import os
import numpy as np

size_im = 256

class RGBtoLAB(object):
    # Convert RGD ndarray in sample to LAB

    def __call__(self, image):
        X = rgb2lab(image)[:, :, 0]
        Y = rgb2lab(image)[:, :, 1:] # selects the two color layers green–red and blue–yellow

        Y = Y / 128 # force the range between -1 and 1
        #sample = {'image_bw': X, 'image_ab': Y}
        #image_bw, image_ab = sample['image_bw'], sample['image_ab']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image_bwT = torch.from_numpy(X).float()
        image_bwT = image_bwT.unsqueeze(dim=0)  # torch.Size([1, 400, 200]) it is like transpose((2, 0, 1))

        image_abT = Y.transpose((2, 0, 1))
        #image_abT = image_ab.transpose((2, 1, 0))
        image_abT = torch.from_numpy(image_abT).float()
        #image_abT = torch.from_numpy(image_ab).float()
        #image_abT = image_abT.unsqueeze(dim=0)

        return {'image_bw': image_bwT, 'image_ab': image_abT}

def buildLABImage(X, Y, output):
    X = X.detach().numpy()
    X = X[0].transpose((1, 2, 0))
    X = np.squeeze(X, axis=2)

    Y = Y.detach().numpy()
    Y = Y[0].transpose((1, 2, 0))
    Y = Y * 128

    output = output.detach().numpy()  # output.numpy().transpose((1, 2, 0))
    output = output[0].transpose((1, 2, 0))
    output = output * 128  # convert to LAB presents -128 _ 128 from -1 _ 1

    # a black RGB canvas
    cur = np.zeros((size_im, size_im, 3))
    cur1 = np.zeros((size_im, size_im, 3))

    cur[:, :, 0] = X  # copy the original BW layer
    cur[:, :, 1] = output[:, :, 0]
    cur[:, :, 2] = output[:, :, 1]

    # checking original image
    cur1[:, :, 0] = X  # copy the original BW layer
    cur1[:, :, 1] = Y[:, :, 0]
    cur1[:, :, 2] = Y[:, :, 1]


    output_img = lab2rgb(cur)
    cur1 = lab2rgb(cur)
    return output_img, cur1


class ImgDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.lst_names = os.listdir(root_dir)
        self.LABtransform = RGBtoLAB()
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

        if self.LABtransform:
            sample = self.LABtransform(sample)

        # check an image or dict with two keys

        return sample

