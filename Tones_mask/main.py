
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import ImgDataset, buildImagefromMask
from model import nn_model, train_model, prediction
from skimage.color import rgb2lab, lab2rgb

import matplotlib.pyplot as plt
import numpy as np


def main():


    '''1.Upload images to Pytorch Loader to custom Dataset class'''
    print("1. Loading train dataset")
    batch_size = 1
    target_img_path = '/home/sveta/DL_projects/roi/film_color/color_tones'



    #for i in range(len(data_aug_train)):
    #    sample = data_aug_train[i]
    #    print(i, sample['image_bw'].shape, sample['image_ab'].shape)

    '''2. Train model'''
    print('2. Trainig the model')
    epochs=100


    # Loss & Optimizer
    loss = torch.nn.MSELoss().type(torch.FloatTensor)  # the content distance
    # Need to figure out what optim should use
    # a loss curve that starts well and after X hundred epochs makes a sudden jump?


    test_img_path = '/home/sveta/DL_projects/roi/film_color/test'
    size_r = 400
    pred_img = np.zeros((size_r, size_r, 3))
    bw_original = np.zeros((size_r, size_r))
    rgb_original = np.zeros((size_r, size_r, 3))

    # for each tone zone
    for i in range(0,3):
        print("Tone area ", i)
        model = nn_model()
        optimizer = optim.RMSprop(model.parameters(), lr=1e-4, weight_decay=0, eps=1e-07)
        data_aug_train = ImgDataset(root_dir=target_img_path, id_tone=i)  # , transform=tfs # for augmentation data
        train_aug_loader = DataLoader(data_aug_train, batch_size=batch_size)
        loss_history = train_model(model,loss, optimizer, train_aug_loader, epochs)

        plt.figure(figsize=(10, 10), dpi=300)
        plt.grid()
        plt.plot(range(0,len(loss_history)), loss_history )
        plt.xlabel('epoch')
        plt.ylabel('loss value')
        plt.savefig("Loss per epoch on the whole set.png")

        '''3. Prediction output colorizations'''
        print('3. Prediction output colorizations')
        data_aug_test = ImgDataset(root_dir=test_img_path, id_tone=i)
        test_aug_loader = DataLoader(data_aug_test)

        X, Y, output = prediction(model, test_aug_loader)  # a bw image as input from the original image
        # buid an image
        X, Y, output = buildImagefromMask(X,Y,output)
        #if i == 0:
        #    pred_img += output
        #else:
        #    pred_img += Y # normalization
        bw_original += X
        rgb_original += Y
        pred_img += output

    cur = np.zeros((400, 400, 3))
    X1 = rgb2lab(Y)[:, :, 0]
    Y = rgb2lab(pred_img)[:, :, 1]
    Z = rgb2lab(pred_img)[:, :, 2]
    cur[:, :, 0] = X1  # X[:,:,0] #X[0][:,:,0] # copy the original BW layer
    cur[:, :, 1] = Y
    cur[:, :, 2] = Z
    As = lab2rgb(cur)
    fig = plt.figure()
    plt.imshow(As)
    fig.savefig('As_img.png')

    '''4. Build the output image'''
    print('4. Save the colorized image')

    fig = plt.figure()
    plt.imshow(bw_original, cmap='gray')
    fig.savefig('bw_img.png')

    fig = plt.figure()
    plt.imshow(rgb_original)
    fig.savefig('original_img.png')

    fig = plt.figure()
    plt.imshow(pred_img)
    fig.savefig('pred_img_test.png')



if __name__ == '__main__':
    main()


