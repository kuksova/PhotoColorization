
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import ImgDataset, buildLABImage
from model import nn_model, train_model, prediction

import matplotlib.pyplot as plt


def main():
    '''1.Upload images to Pytorch Loader to custom Dataset class'''
    print("1. Loading train dataset")
    batch_size = 1
    target_img_path = './images/color'
    data_aug_train = ImgDataset(root_dir=target_img_path) # , transform=tfs # for augmentation data
    train_aug_loader = DataLoader(data_aug_train, batch_size=batch_size)


    #for i in range(len(data_aug_train)):
    #    sample = data_aug_train[i]
    #    print(i, sample['image_bw'].shape, sample['image_ab'].shape)

    '''2. Train model'''
    print('2. Trainig the model')
    epochs=100
    model = nn_model()

    # Loss & Optimizer
    loss = torch.nn.MSELoss().type(torch.FloatTensor)  # the content distance
    # Need to figure out what optim should use
    # a loss curve that starts well and after X hundred epochs makes a sudden jump?
    optimizer = optim.RMSprop(model.parameters(), lr=1e-4, weight_decay=0, eps=1e-07)

    loss_history = train_model(model,loss, optimizer, train_aug_loader, epochs)

    plt.figure(figsize=(10, 10), dpi=300)
    plt.grid()
    plt.plot(range(0,len(loss_history)), loss_history )
    plt.xlabel('epoch')
    plt.ylabel('loss value')
    plt.savefig("Loss per epoch on the whole set.png")

    '''3. Prediction output colorizations'''
    print('3. Prediction output colorizations')
    test_img_path = './images/test/'
    data_aug_test = ImgDataset(root_dir=test_img_path)
    test_aug_loader = DataLoader(data_aug_test)

    X, Y, output = prediction(model, test_aug_loader)  # a bw image as input from the original image

    '''4. Build the output image'''
    print('4. Save the colorized image')
    output_img, cur1 = buildLABImage(X,Y,output)

    fig = plt.figure()
    plt.imshow(output_img)
    fig.savefig('colorized_img.png')

    fig = plt.figure()
    plt.imshow(cur1)
    fig.savefig('original_test_img.png')



if __name__ == '__main__':
    main()


