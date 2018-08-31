
# Torch imports
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

# Other imports
import os
import pdb
import imageio
import matplotlib
import numpy as np
matplotlib.use('Agg')
from others import temp
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST




os.system('clear')
os.environ["CUDA_VISIBLE_DEVICES"]="2"

#-------------------------------------------------------------------------
# Checking compatibility

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, 
            requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, 
            size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

#-------------------------------------------------------------------------
# Parameters

BATCH_SIZE = 2048
lr = 0.0001
momentum = 0.9
n_epochs = 500
noise_level = 0.05
mkimage = True
ROOT_MNIST = './dataset'
LOSS_PATH = '../results'




join = os.path.join
MNIST_db = MNIST(root = ROOT_MNIST,train = True, download = True, transform=torchvision.transforms.ToTensor())
train_loader = Data.DataLoader(dataset=MNIST_db, batch_size=BATCH_SIZE, shuffle=True)
total = MNIST_db.__len__()

name = 'RL_'+str(lr)+'_'+str(n_epochs)+'_'+str(BATCH_SIZE)+'.png'

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
        noisy = im.clone() + (torch.rand(28*28)<self.noise_level).float()
        noisy = noisy*((noisy<1).float()) + (noisy>=1).float()
        #print(noisy)
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

device = toch.device('cpu')
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train = Data.DataLoader(dataset = Noisy_MNIST(noise_level = noise_level),batch_size = BATCH_SIZE, shuffle = True)

model = AutoEncoder(features = 32)
loss_function = nn.MSELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
model.to(device)


plotloss = [0 for _ in range(n_epochs)]
timer = temp.Timer()

for epoch in range(n_epochs):
    running_loss = 0
    temp.print_message(epoch, timer, n_epochs)
    for idx, dicc in enumerate(train):
        images = dicc['noisy'].to(device, dtype = torch.float)
        label = dicc['image'].to(device, dtype = torch.float)
        _, image = model(images,BATCH_SIZE)
        loss = loss_function(image,label)
        running_loss += loss.item()#/float(BATCH_SIZE)
        loss.backward() 
        optimizer.step()
        if (idx)%(total//(BATCH_SIZE*10)) == 0 or idx == total//BATCH_SIZE-1:
            print('Process: {:.4f}'.format((idx+1)*BATCH_SIZE/total),'% | Running loss: {:.4f}'.format( running_loss))
            if mkimage:
                # take first image
                picture = (255*image[1,:]).view(28,28).to('cpu').detach().numpy().astype(np.uint8)
                orig = (255*images[1,:]).view(28,28).to('cpu').detach().numpy().astype(np.uint8)
                imageio.imwrite(join(LOSS_PATH,str(epoch)+'_'+str(idx)+'.png'),np.concatenate((orig,picture), axis = 1))
    print('Total loss: {:.6}'.format(running_loss))





    plotloss[epoch] = running_loss



fig = plt.figure()
plt.plot([i+1 for i in range(len(plotloss))], plotloss, 'r')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(join(LOSS_PATH,name))
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
        




