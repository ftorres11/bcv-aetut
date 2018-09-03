import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, enc = [[28*28,128],[128,64],[64,12],[64,3]], dec = [[3,12],[12,64],[64,128],[128,28*28]]):
        super(Autoencoder,self).__init__()
        self.len = len(enc)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(len(enc)):
            self.encoder.append(layer(in_features = enc[i][0], out_features = enc[i][1], finish = i == self.len-1))
            self.decoder.append(layer(in_features = dec[i][0], out_features = dec[i][1], finish = i == self.len-1))
    def forward(self,x):
        for i in range(self.len):
            x = self.encoder[i](x)
        for i in range(self.len):
            x = self.decoder[i](x)
        return x




class layer(nn.Module):
    def __init__(self,in_features,out_features,finish = False):
        super(layer,self).__init__()
        self.layer = nn.Linear(in_features,out_features)
        self.finish = finish
        if finish:
            self.relu = nn.ReLU()
    def forward(self,x):
        x = self.layer(x)
        if self.finish:
            x = self.relu(x)
        return x


class VarationalAutoEncoder(nn.Module):
    def __init__(self, features = 10):
        super(VarationalAutoEncoder,self).__init__()
        self.features = features
        self.CuttedEncoder = nn.Sequential(
            nn.Linear(in_features = 28*28, out_features = 256),
            nn.ReLU(),
            nn.Linear(in_features = 256, out_features = 64),
            nn.ReLU()
            )

        self.LogVariance = nn.Linear(in_features = 64, out_features = features)
        self.Mu = nn.Linear(in_features = 64, out_features = features)

        self.Decoder = nn.Sequential(
            nn.Linear(in_features = features, out_features = 64),
            nn.ReLU(),
            nn.Linear(in_features = 64, out_features = 256),
            nn.ReLU(),
            nn.Linear(in_features = 256, out_features = 28*28),
            nn.ReLU()
            )

    def forward(self,x):
        x = self.CuttedEncoder(x)

        # Reparametrization
        mu = self.Mu(x)
        std = self.LogVariance(x) #torch.exp(0.5*self.LogVariance(x))
        x = mu + std*(torch.randn(x.shape[0],self.features).to(device))

        x = self.Decoder(x)

        return x, mu, std

class VAE_loss(nn.Module):
    def __init__(self):
        super(VAE_loss,self).__init__()
        self.reconstruction_loss = nn.MSELoss()
    def forward(self, image, label, mu, variance):
        R_L = self.reconstruction_loss(image,label)
        mDKL = -0.5*torch.sum(1.+torch.log(variance*variance)-mu*mu-variance*variance)
        return R_L+mDKL
