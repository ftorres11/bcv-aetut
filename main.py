
# Torch imports
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

# Other imports
import os
import pdb
import shutil
import imageio
import argparse
import matplotlib
import numpy as np
matplotlib.use('Agg')
from others import temp
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST

import lib.encoders as auto_gen
from lib.data import *

os.system('clear')

#-------------------------------------------------------------------------
# Default definitions
join = os.path.join
ROOT_MNIST = 'dataset'
LOSS_PATH = '../results'

#-------------------------------------------------------------------------
# Parser Arguments
parser = argparse.ArgumentParser(description='Train an autoencoder')
parser.add_argument('--learning_rate',
    help='Learning rate of the algorythm.', dest = 'lr',
    default=0.00001, type=float, required = True)

parser.add_argument('--epochs', help='Number of epochs', dest = 'epochs',
    default=750, type=int, required = True)

parser.add_argument('--batch_size', help='Batch size pero iteration', 
    dest = 'batch_size', default=512, type=int, required = True)

parser.add_argument('--noise_level', help='Batch size per iteration', 
    dest = 'noise', default=0.05, type=float, required = True)
parser.add_argument('--mkimage', help='{} {}'.format('Generate images',
    'randomly every 10 epochs'), dest = 'mkimage', default = True, 
     type=bool,required = False)
parser.add_argument('--dest', help='Destination to store output images',
    dest='name', default='experiment', type = str, required = False)
args = parser.parse_args()

#-------------------------------------------------------------------------
# Extracting parameters from the parser
BATCH_SIZE = args.batch_size
lr = args.lr
n_epochs = args.epochs
noise_level = args.noise
mkimage = args.mkimage
name = args.name
exp_route = join(LOSS_PATH,name)

#-------------------------------------------------------------------------
# Loading data
MNIST_db = MNIST(root = ROOT_MNIST,train = True, download = True, 
    transform=torchvision.transforms.ToTensor())
train_loader = Data.DataLoader(dataset=MNIST_db, batch_size=BATCH_SIZE, 
    shuffle=True)
total = MNIST_db.__len__()

#-------------------------------------------------------------------------
# Checking existance of target directory
if not os.path.exists(exp_route):
    os.makedirs(exp_route)
else:
    shutil.rmtree(exp_route)
    os.makedirs(exp_route)
   
#-------------------------------------------------------------------------
# Allocating devices and allocating sets
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train = Data.DataLoader(dataset = Noisy_MNIST(data_root = ROOT_MNIST,
    noise_level = noise_level), batch_size = BATCH_SIZE, shuffle = True)

#-------------------------------------------------------------------------
# Loading Model and functions
model = auto_gen.Autoencoder(enc = [[28*28,256],[256,64],[64,10]], 
    dec = [[10,64],[64,256],[256,28*28]])
loss_function = nn.MSELoss().to(device)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
model.to(device)

plotloss = [0 for _ in range(n_epochs)]
timer = temp.Timer()

#-------------------------------------------------------------------------
# Training routine
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
        # Plots
        if (idx)%(total//(BATCH_SIZE*10)) == 0 or idx == total//BATCH_SIZE-1:
            print('Process: {:.4f}'.format((idx+1)*BATCH_SIZE/total),
                '% | Running loss: {:.4f}'.format( running_loss))
            if mkimage and torch.rand(1)<0.05:
                # take first image
                pic_prim = (255*image[1,:]).view(28,28).to('cpu')
                picture = pic_prim.detach().numpy().astype(np.uint8)
                orig_prim = (255*images[1,:]).view(28,28).to('cpu')
                orig = orig_prim.detach().numpy().astype(np.uint8)
                imageio.imwrite(join(exp_route,
                    '{}_{}.png'.format(str(epoch),str(idx))),
                    np.concatenate((orig,picture), axis = 1))
    print('Total loss: {:.6}'.format(running_loss))
    plotloss[epoch] = running_loss


torch.save(model.state_dict(), join(exp_route,'model.pkl'))

fig = plt.figure()
plt.plot([i+1 for i in range(len(plotloss))], plotloss, 'r')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(join(exp_route,'loss.pdf'))
plt.close(fig)

