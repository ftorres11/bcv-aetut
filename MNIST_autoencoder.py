import torch
import torch.nn as nn
import torch.utils.data as Data
from torchvision.datasets import MNIST
import torchvision


BATCH_SIZE = 64
ROOT_MNIST = './dataset'

MNIST_db = MNIST(root = ROOT_MNIST,train = True, download = True, transform=torchvision.transforms.ToTensor())
train_loader = Data.DataLoader(dataset=MNIST_db, batch_size=BATCH_SIZE, shuffle=True)


for idx, ims in enumerate(train_loader):
	print(ims[0].shape, ims[1].shape)


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





