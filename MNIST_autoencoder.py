
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
import argparse
import matplotlib
import numpy as np
matplotlib.use('Agg')
from others import temp
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST

import encoders.model1 as AE_gen


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

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Train an autoencoder network')

    # Epoch
    parser.add_argument(
        '--learning_rate',
        help='Learning rate of the algoryth.', dest = 'lr',
        default=0.00001, type=float, required = True)

    parser.add_argument(
        '--epochs',
        help='Number of epochs', dest = 'epochs',
        default=750, type=int, required = True)

    parser.add_argument(
        '--batch_size',
        help='Batch size pero iteration', dest = 'batch_size',
        default=512, type=int, required = True)

    parser.add_argument(
        '--noise_level',
        help='Batch size pero iteration', dest = 'noise',
        default=0.05, type=float, required = True)

    return parser.parse_args()

args = parse_args()


BATCH_SIZE = args.batch_size
lr = args.lr
n_epochs = args.epochs
noise_level = args.noise

mkimage = True
ROOT_MNIST = './dataset'
LOSS_PATH = '../results'


join = os.path.join
MNIST_db = MNIST(root = ROOT_MNIST,train = True, download = True, transform=torchvision.transforms.ToTensor())
train_loader = Data.DataLoader(dataset=MNIST_db, batch_size=BATCH_SIZE, shuffle=True)
total = MNIST_db.__len__()

name = 'VAE_Loss_RL_lr'+str(lr)+'_e'+str(n_epochs)+'_bs'+str(BATCH_SIZE)+'_n'+str(noise_level) + '.png'# Plot loss
if os.path.exists(join(LOSS_PATH,name[:-4])):
    os.system('rm -r '+join(LOSS_PATH,name[:-4]))

os.mkdir(join(LOSS_PATH,name[:-4]))

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
        im.requires_grad = False
        label = item[1]
        noisy = im.clone() + (torch.rand(28*28)<self.noise_level).float()
        noisy = noisy*((noisy<1).float()) + (noisy>=1).float()
        #print(noisy)
        return {'image':im, 'noisy':noisy, 'label':label}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train = Data.DataLoader(dataset = Noisy_MNIST(noise_level = noise_level),batch_size = BATCH_SIZE, shuffle = True)

model = AE_gen.Autoencoder(enc = [[28*28,256],[256,64],[64,10]], dec = [[10,64],[64,256],[256,28*28]])
loss_function = nn.MSELoss().to(device)

optimizer = optim.Adam(
model.parameters(), lr=lr, weight_decay=1e-5)
model.to(device)


plotloss = [0 for _ in range(n_epochs)]
timer = temp.Timer()

for epoch in range(n_epochs):
    running_loss = 0
    temp.print_message(epoch, timer, n_epochs)

    for idx, dicc in enumerate(train):
        images = dicc['noisy'].to(device, dtype = torch.float)
        label = dicc['image'].to(device, dtype = torch.float)
        # ================== FORWARD =================
        #_, image = model(images)
        image = model(images)
        loss = loss_function(image,label)
        # ================= BACKWARD =================
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()

        running_loss += loss.item()#/float(BATCH_SIZE)
        if (idx)%(total//(BATCH_SIZE*10)) == 0 or idx == total//BATCH_SIZE-1:
            print('Process: {:.4f}'.format((idx+1)*BATCH_SIZE/total),'% | Running loss: {:.4f}'.format( running_loss))
            if mkimage and torch.rand(1)<0.05:
                # take first image
                picture = (255*image[1,:]).view(28,28).to('cpu').detach().numpy().astype(np.uint8)
                orig = (255*images[1,:]).view(28,28).to('cpu').detach().numpy().astype(np.uint8)
                imageio.imwrite(join(LOSS_PATH,name[:-4],str(epoch)+'_'+str(idx)+'.png'),np.concatenate((orig,picture), axis = 1))
    print('Total loss: {:.6}'.format(running_loss))
    plotloss[epoch] = running_loss


torch.save(model.state_dict(), join(LOSS_PATH,name[:-4],'model.pkl'))

fig = plt.figure()
plt.plot([i+1 for i in range(len(plotloss))], plotloss, 'r')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(join(LOSS_PATH,name))
plt.close(fig)

