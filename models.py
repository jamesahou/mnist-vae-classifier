import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args
        self.convolution1 = nn.Sequential(nn.Conv2d(in_channels=self.args.channels, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.convolution2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())

    def forward(self, x):
        x = self.convolution1(x)
        x = self.convolution2(x)
        return x


class Sampler(nn.Module):
    def __init__(self, args):
        super(Sampler, self).__init__()
        self.args = args
        self.z_mean = nn.Linear(128*7*7, self.args.zdim)
        self.z_var = nn.Linear(128*7*7, self.args.zdim)

    def sample_z(self, mean, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mean)
        else:
            return mean

    def forward(self, x):
        x = x.view(-1, 128*7*7)
        mean = self.z_mean(x)
        var = self.z_var(x)
        z = self.sample_z(mean, var)
        return z, mean, var


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.args = args
        self.expand = nn.Linear(self.args.zdim, 128*7*7)
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(in_channels=64, out_channels=self.args.channels, kernel_size=4, stride=2, padding=1))
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, z):
        out = self.expand(z)
        out = out.view(out.size(0), 64*2, 7, 7) 
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = torch.sigmoid(out)
        return out


class Classifier(nn.Module):
    def __init__(self, args, out_c=10):
        super(Classifier, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(128*7*7, 10)

    def forward(self, z):
        out = self.fc1(z)
        return out

class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.args = args
        self.encoder = Encoder(self.args)
        self.sampler = Sampler(self.args)
        self.decoder = Decoder(self.args)
        self.classifier = Classifier(self.args)

    def display(self):
        print(self)

    def forward(self, x):
        latent = self.encoder(x)
        c = self.classifier(latent.view(-1, 128*7*7))
        z, mean, var = self.sampler(latent)
        out = self.decoder(z)
        return out, c, mean, var


class CNN(nn.Module):
    def __init__(self, args, out_c=10):
        super(CNN, self).__init__()
        self.args = args
        self.extractor = Encoder(self.args)
        self.classifier = Classifier(self.args)
    
    def forward(self, x):
        out = self.extractor(x)
        out = out.view(-1, 128*7*7)
        out = self.classifier(x)
        return out

if __name__ == "__main__":
    a = VAE().display()
