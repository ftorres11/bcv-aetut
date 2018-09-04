import torch
import matplotlib
import torchvision
matplotlib.use('Agg')
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST

#-------------------------------------------------------------------------
# Classes
#-------------------------------------------------------------------------
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
#-------------------------------------------------------------------------
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
#-------------------------------------------------------------------------
class Noisy_MNIST():
    def __init__(self, ROOT_MNIST, n_level):
        MNIST_db = MNIST(root = ROOT_MNIST,train = True, download = True, 
            transform=torchvision.transforms.ToTensor())
        self.getitem = MNIST_db.__getitem__
        self.len = MNIST_db.__len__()
        self.noise_level = n_level

    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        item = self.getitem(idx)
        im = item[0].view(-1)
        im.requires_grad = False
        label = item[1]
        noisy = im.clone() + (torch.rand(28*28)<self.noise_level).float()
        noisy = noisy*((noisy<1).float()) + (noisy>=1).float()
        return {'image':im, 'noisy':noisy, 'label':label}

