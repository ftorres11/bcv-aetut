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
# Class Definition
#-------------------------------------------------------------------------
class Noisy_MNIST():
    def __init__(self, data_root, noise_level):
        MNIST_db = MNIST(root = data_root,train = True, download = True, 
            transform=torchvision.transforms.ToTensor())
        self.getitem = MNIST_db.__getitem__
        self.len = MNIST_db.__len__()
        self.noise_level = noise_level

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

