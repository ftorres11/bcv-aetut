import os
import pdb
import torch
import argparse
import matplotlib
import torchvision
matplotlib.use('Agg')
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST

#-------------------------------------------------------------------------
parser = argparse.ArgumenParser(description='Autoencoder trained on MNIST')
parser.add_argument('--bsz',type=int,default=64,help='Batch size')
parser.add_argument('--lr',type=float,default=1e-4,help='Learning rate')
parser.add_argument('--ep',type=int,default=100,help='Number of Epochs')

BATCH_SIZE = 1024
lr = 0.000001
momentum = 0.9
n_epochs = 100
noise_level = 0
ROOT_MNIST = './dataset'
LOSS_PATH = '.'
join = os.path.join
MNIST_db = MNIST(root = ROOT_MNIST,train = True, download = True, transform=torchvision.transforms.ToTensor())
train_loader = Data.DataLoader(dataset=MNIST_db, batch_size=BATCH_SIZE, shuffle=True)


class Noisy_MNIST():
    def __init__(self, noise_level = noise_level):
        MNIST_db = MNIST(root = ROOT_MNIST,train = True, download = True, transform=torchvision.transforms.ToTensor())
        self.getitem = MNIST_db.__getitem__
        self.len = MNIST_db.__len__()
        self.noise_level = noise_level

    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        item = self.getitem(idx)
        im = item[0].view(-1)
        label = item[1]
        noisy = im.clone() + torch.rand(28*28)<self.noise_level
        noisy = noisy*(noisy<1) + noisy>=1
        return {'image':im, 'noisy':noisy, 'label':label}



class AutoEncoder(nn.Module):
    def __init__(self, features = 32):
        super(AutoEncoder,self).__init__()
        self.Encoder = nn.Sequential(
            nn.Linear(in_features = 28*28, out_features = 512),
            nn.ReLU(),
            nn.Linear(in_features = 512, out_features = 128),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features = features),
            nn.ReLU()
            )
        self.Decoder = nn.Sequential(
            nn.Linear(in_features = features, out_features = 128),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features = 512),
            nn.ReLU(),
            nn.Linear(in_features = 512, out_features = 28*28),
            nn.ReLU()
            )

    def forward(self,x, batch_size):
        Enc = self.Encoder(x)
        Dec = self.Decoder(Enc)

        return Enc, Dec

class VarationalAutoEncoder(nn.Module):
    def __init__(self, features = 32):
        super(VarationalAutoEncoder,self).__init__()

        self.CuttedEncoder = nn.Sequential(
            nn.Linear(in_features = 28*28, out_features = 512),
            nn.ReLU(),
            nn.Linear(in_features = 512, out_features = 128),
            nn.ReLU()
            )

        self.Variance = nn.Linear(in_features = 128, out_features = features)
        self.Mu = nn.Linear(in_features = 128, out_features = features)

        self.Decoder = nn.Sequential(
            nn.Linear(in_features = features, out_features = 128),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features = 512),
            nn.ReLU(),
            nn.Linear(in_features = 512, out_features = 28*28),
            nn.ReLU()
            )



    def forward(self,x, batch_size):
        x = self.CuttedEncoder(x)
        x = self.Mu(x) + self.Variance(x)*torch.randn(x.shape)
        x = self.Decoder(x)

        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train = Data.DataLoader(dataset = Noisy_MNIST(noise_level = noise_level),batch_size = BATCH_SIZE, shuffle = True)
model = AutoEncoder(features = 32)
loss_function = nn.MSELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
model.to(device)
plotloss = [0 for _ in range(n_epochs)]

for epoch in range(n_epochs):
    running_loss = 0
    print('Epoch:',epoch+1)
    for idx, dicc in enumerate(train):
        image = dicc['noisy'].to(device, dtype = torch.float)
        label = dicc['image'].to(device, dtype = torch.float)
        _, image = model(image,BATCH_SIZE)
        loss = loss_function(image,label)
        running_loss += loss.item()/float(BATCH_SIZE)
        loss.backward() 
        optimizer.step()

        if (idx/2)%71 == 0:
            print('Running loss:', running_loss)

    plotloss[epoch] = running_loss



fig = plt.figure()
plt.plot([i+1 for i in range(len(plotloss))], plotloss, 'r')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(join(LOSS_PATH,'running_loss.png'))
plt.close(fig)

'''
class Autoencoder(nn.Module):
    def __init__(self, dim_rep = 32, channels = 16):
        super(AutoEncoder, self).__init__()
        self.Encoder = Encoder(dim_rep = dim_rep, channels = channels)
        self.Decoder = Decoder(dim_rep = dim_rep, channels = channels)
    def forward(self,x):
        Enc = self.Encoder(Dec)
        Dec = self.Decoder(Enc)
        return Enc, Dec

class Encoder(nn.Module):
    def __init__(self, dim_rep = 32, channels = 16):
        super(Encoder,self).__init__()
        self.cnn1 = nn.Conv2d(in_channels = 1, out_channels = channels, kernel_size = 3, padding = 1) # 28x28-> 28x28
        self.cnn2 = nn.Conv2d(in_channels = 2*channels, out_channels = 4*channels, kernel_size = 3, padding = 1) # 14x14 -> 14x14
        self.cnn3 = nn.Conv2d(in_channels = 4*channels, out_channels = dim_rep, kernel_size = 13) # 7x7 -> 1x1

        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(kernel_size = 2)

    def forward(self,x):
        x = self.mp(self.relu(self.cnn1(x)))
        x = self.mp(self.relu(self.cnn2(x)))
        x = self.relu(self.cnn3(x))

class Decoder(nn.Module):
    def __init__(self, dim_rep = 32, channels = 16):
        super(Decoder,self).__init__()
'''
        



