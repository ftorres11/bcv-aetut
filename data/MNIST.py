import torch
import torch.nn as nn
from torchvision.datasets import MNIST
#-------------------------------------------------------------------------
# Class Definition
#-------------------------------------------------------------------------
class Noisy_MNIST():
    def __init__(self, noise_level = noise_level):
        MNIST_db = MNIST(root = ROOT_MNIST,train = True, download = True, 
            transform=torchvision.transforms.ToTensor())
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
