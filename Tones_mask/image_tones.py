import os

import skimage.color
import torch
import numpy as np
from skimage import io, transform
from skimage.color import rgb2lab, lab2rgb, rgba2rgb
import pandas as pd

import cv2

import matplotlib.pyplot as plt


import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import transforms

path2 = '/home/sveta/DL_projects/roi/film_color/color_tones'

# A histogram is a representation of the number of pixels in a photo at each luminance percentage.
def hist_chanel(image):
    # tuple to select colors of each channel line
    colors = ("red", "green", "blue")
    channel_ids = (0, 1, 2)

    # create the histogram plot, with three lines, one for
    # each color
    plt.figure()
    plt.xlim([0, 256])
    for channel_id, c in zip(channel_ids, colors):
        histogram, bin_edges = np.histogram(
            image[:, :, channel_id], bins=256, range=(0, 256)
        )
        plt.plot(bin_edges[0:-1], histogram, color=c)

    plt.title("Color Histogram")
    plt.xlabel("Color value")
    plt.ylabel("Pixel count")

    plt.savefig('img_tones_hist.png')


for img_name2 in os.listdir(path2):
    image2 = io.imread(os.path.join(path2, img_name2))  # numpy array
    # image = rgba2rgb(image_rgba)
    #image = transform.resize(image2, (400, 200))
    #RGB_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    #image2[image2 < 2 * 255/ 3] = 0  # keep highlights
    fig = plt.figure()
    plt.imshow(image2)
    fig.savefig('input_image.png')

    hist_chanel(image2) # cool hist function

# 4. Build the output image
cur = np.zeros((image2.shape[0], image2.shape[1], 3))

cur[:,:,0] = image2[:, :, 0]
cur[:,:,1] = image2[:, :, 1]
cur[:,:,2] = image2[:, :, 2]-50.0
#fig = plt.figure()
#plt.imshow((cur))#.astype(np.uint8))
#fig.savefig('img_output.png')


# Light segmentation
# 5. thresholds for the masks
# refer to hue histogram and input image is bw
threshold = 1.0/3
size_im = 400
image2 = transform.resize(image2, (size_im, size_im))
dark_tones_mask = image2 < threshold
mid_tones_mask = (threshold < image2) & (image2 < 2*threshold)
light_tones_mask = image2 > 2*threshold

dart_tones = image2 * dark_tones_mask
mid_tones = image2 * mid_tones_mask
light_tones_mask = image2 * light_tones_mask

img_out = dart_tones + mid_tones + light_tones_mask

gray_image = skimage.color.rgb2gray(image2)
gray_dark_tones_mask = gray_image < threshold
gray_tones = gray_image * gray_dark_tones_mask

fig = plt.figure()
plt.imshow((dart_tones))#.astype(np.uint8))
fig.savefig('img_dark.png')

fig = plt.figure()
plt.imshow((img_out))#.astype(np.uint8))
fig.savefig('img_out.png')

fig = plt.figure()
plt.imshow(gray_tones, cmap='gray')#.astype(np.uint8))
fig.savefig('gray_dark.png')

dark_tones_mask_red = image2[:,:,0] < threshold
dark_tones_red = image2[:,:,0] * dark_tones_mask_red
#dark_tones_mask_green = image2[:,:,1] < threshold
#dark_tones_green = image2[:,:,1] * dark_tones_mask_green
#print(np.array_equal(dark_tones_mask_red, dark_tones_mask_green))

fig = plt.figure()
plt.imshow((dark_tones_red))#.astype(np.uint8))
fig.savefig('img_dark_red.png')



#fig = plt.figure()
#cur = np.zeros((image2.shape[0], image2.shape[1], 3))
#cur[:,:,0] = image2[:, :, 0]

#plt.imshow(image2)
#fig.savefig('img_red.png')
"""
cur = np.zeros((image2.shape[0], image2.shape[1], 3))
#cur[:,:,0] = image2[:, :, 0]
#image2 = cur


dart_tones = image2 * dark_tones_mask
fig = plt.figure()
plt.imshow((dart_tones))#.astype(np.uint8))
fig.savefig('img_dark.png')

light_tones = image2 * light_tones_mask
fig = plt.figure()
plt.imshow((light_tones))#.astype(np.uint8))
fig.savefig('img_light.png')

mid_tones = image2 * mid_tones_mask
fig = plt.figure()
plt.imshow((mid_tones))#.astype(np.uint8))
fig.savefig('img_mid.png')


mask = dart_tones+light_tones+mid_tones # final image
fig = plt.figure()
plt.imshow((mask))#.astype(np.uint8))
fig.savefig('img_all_masks.png')

#mask = upper_mask * lower_mask * saturation_maskred = bags[:, :, 0] * mask
#green = bags[:, :, 1] * mask
#blue = bags[:, :, 2] * mask
#bags_masked = np.dstack((red, green, blue))
#imshow(bags_masked)

# 6. Training and Prediciton
"""






