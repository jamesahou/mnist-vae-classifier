import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

"""
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
"""
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args
        resnet = models.resnet18(pretrained=False)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(resnet.fc.in_features, 1024)
        self.bn1 = nn.BatchNorm1d(1024, momentum=0.01)
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024, momentum=0.01)
    
    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.bn1(self.fc1(x))
        x = self.relu(x)
        x = self.bn2(self.fc2(x))
        x = self.relu(x)
        return x

class Sampler(nn.Module):
    def __init__(self, args):
        super(Sampler, self).__init__()
        self.args = args
        #self.z_mean = nn.Linear(128*7*7, self.args.zdim)
        #self.z_var = nn.Linear(128*7*7, self.args.zdim)
        self.z_mean = nn.Linear(1024, self.args.zdim)
        self.z_var = nn.Linear(1024, self.args.zdim)

    def sample_z(self, mean, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mean)
        else:
            return mean

    def forward(self, x):
        #x = x.view(-1, 128*7*7)
        mean = self.z_mean(x)
        var = self.z_var(x)
        z = self.sample_z(mean, var)
        return z, mean, var


"""class Decoder(nn.Module):
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
"""
class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.args = args
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)      # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding
        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=self.k4, stride=self.s4,
                               padding=self.pd4),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=self.k3, stride=self.s3,
                               padding=self.pd3),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2),
            nn.BatchNorm2d(3, momentum=0.01),
            nn.Sigmoid()    # y = (y1, y2, y3) \in [0 ,1]^3
        )

        self.fc4 = nn.Linear(self.args.zdim, 1024)
        self.fc_bn4 = nn.BatchNorm1d(1024)
        self.fc5 = nn.Linear(1024, 64 * 4 * 4)
        self.fc_bn5 = nn.BatchNorm1d(64 * 4 * 4)
        self.relu = nn.ReLU()

    def forward(self, z):
        x = self.relu(self.fc_bn4(self.fc4(z)))
        x = self.relu(self.fc_bn5(self.fc5(x))).view(-1, 64, 4, 4)
        x = self.convTrans6(x)
        x = self.convTrans7(x)
        x = self.convTrans8(x)
        x = F.interpolate(x, size=(32, 32), mode='bilinear')
        return x

class Classifier(nn.Module):
    def __init__(self, args, out_c=10):
        super(Classifier, self).__init__()
        self.args = args
        #self.fc1 = nn.Linear(128*7*7, 10)
        self.fc1 = nn.Linear(1024, 10)

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
        #c = self.classifier(latent.view(-1, 128*7*7))
        c = self.classifier(latent)
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
        out = self.classifier(out)
        return out

if __name__ == "__main__":
    a = VAE().display()
