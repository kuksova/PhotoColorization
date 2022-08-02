import torch
import torch.nn as nn
import torch.optim as optim

# 2. Build a model
def nn_model():
    nn_model = nn.Sequential(
                nn.Conv2d(1, 8, 3, padding=1, stride=2),
    nn.ReLU(inplace=True),
                nn.Conv2d(8, 8, 3, padding='same'),
    nn.ReLU(inplace=True),
                nn.Conv2d(8, 16, 3, padding='same'),
    nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, 3, padding=1, stride=2),
    nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 3, padding='same'),
    nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1, stride=2),
    nn.ReLU(inplace=True),
    nn.UpsamplingNearest2d(scale_factor=2),
                nn.Conv2d(32, 16, 3, padding='same'),
    nn.ReLU(inplace=True),
    nn.UpsamplingNearest2d(scale_factor=2),
                nn.Conv2d(16, 8, 3, padding='same'),
    nn.ReLU(inplace=True),
    nn.UpsamplingNearest2d(scale_factor=2),
                nn.Conv2d(8, 2, 3, padding='same'),
    nn.Tanh() )

    nn_model.type(torch.FloatTensor)
    return nn_model


def train_model(model, loss, optimizer, train_loader, num_epochs):
    loss_history = []

    for epoch in range(num_epochs):
        model.train()  # Enter train mode

        loss_accum = 0

        for i_step, sample_batched in enumerate(train_loader):
            x = sample_batched['image_bw']
            y = sample_batched['image_ab'].type(torch.FloatTensor)
            #print(x.size())
            prediction = model(x)

            loss_value = loss(prediction, y)
            optimizer.zero_grad()
            loss_value.backward()

            optimizer.step()  # update weights

            loss_accum += loss_value

        ave_loss = loss_accum / (i_step + 1)
        loss_history.append(float(ave_loss))

        print("Average loss in batch per epoch: %f" % (ave_loss))
    return loss_history

def prediction(model, loader):

    model.eval()

    for i_step, sample_batched in enumerate(loader):
        x = sample_batched['image_bw']
        y = sample_batched['image_ab'].type(torch.FloatTensor)
        prediction_Y = model(x) # convert from pytorch tensor

    return x, y, prediction_Y