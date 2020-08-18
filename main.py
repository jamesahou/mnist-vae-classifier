import argparse
import torch.optim as optim
import torch.nn as nn
from models import CNN, VAE
from torchvision import transforms, datasets
import torchvision
import torch
from torch.autograd import Variable
from torchvision.utils import save_image
import torch.nn.functional as F

def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def config():
    parser = argparse.ArgumentParser(description="MNIST VAE CLASSIFIER")
    parser.add_argument('--channels', default=1, type=int)
    parser.add_argument('--cuda', default=True, type=str2bool)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--zdim', default=2, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--model', default='vae', type=str)
    parser.add_argument('--test', default=True, type=str2bool)
    parser.add_argument('--train', default=True, type=str2bool)
    parser.add_argument('--directory', default='samples/sample.png', type=str)
    args = parser.parse_args()
    return args

def vae_loss(inputs, outputs):
#def vae_loss(inputs, outputs, mean, logvar):
    recon_loss_fc = nn.MSELoss()
    recon_loss = recon_loss_fc(inputs, outputs)
    #recon_loss = F.binary_cross_entropy(outputs.view(-1, 784), inputs.view(-1, 784), reduction='sum')
    #kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss

#def t_loss(inputs, outputs, c, mean, var, labels):
#def t_loss(inputs, outputs, mean, var, labels):
def t_loss(inputs, outputs):
    #classification_criterion = nn.CrossEntropyLoss()
    #classification_loss = classification_criterion(c, labels)
    recon_loss = vae_loss(inputs, outputs)
    #return classification_loss + vae_l, kl_loss, recon_loss, classification_loss
    return recon_loss

def train_vae(model, train_loader, device, args):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    total_step = len(train_loader)
    acc_list = []
    train_loss_avg = []
    for epoch in range(args.epoch):
        loss_tracker = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            
            inputs = Variable(inputs.to(device))
            labels = Variable(labels.to(device))
            optimizer.zero_grad()
            #outputs, classification, mean, var = model(inputs)
            outputs = model(inputs)
            """
            print("Input ", inputs)
            print("Output ", outputs)"""
            #loss, kl_loss, recon_loss, classification_loss = t_loss(inputs, outputs, classification, mean, var, labels)
            #loss = t_loss(inputs, outputs)
            loss = torch.mean((inputs-outputs)**2)
            loss.backward()
            """for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    print(name, torch.sum(parameter.grad))
                else:
                    print(name, "gradient is none")
            """
            optimizer.step()
            loss_tracker += loss.item()
            """total = labels.size(0)
            _, predicted = torch.max(classification.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)
            """
            if (i+1) % 100 == 0:
                #print("Epoch: [%d/%d], Step [%d/%d], Running Loss: %.4f Loss: %.4f, Classification Accuracy: %.2f Reconstruction Loss: %.4f KL Loss: %.4f Class Loss: %.4f" %(epoch, args.epoch,
                #    i+1, total_step, loss_tracker//100, loss.item(), correct/total*100, recon_loss.item(), kl_loss.item(), classification_loss.item()))
                print("Epoch: [%d/%d], Step [%d/%d], Loss: %.4f" % (epoch, args.epoch, i+1, total_step, loss.item()))

def train_classifier(model, train_loader, device, args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fc = nn.CrossEntropyLoss()
    acc_list = []
    total_step = len(train_loader)
    for epoch in range(args.epoch):
        loss_tracker = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = Variable(inputs.to(device))
            labels = Variable(labels.to(device))
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fc(outputs, labels)
            loss.backward()
            loss_tracker += loss.item()
            optimizer.step()
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)
            if (i+1) % 100 == 0:
                print("Epoch: [%d/%d], Step [%d/%d], Running Loss: %.4f Loss: %.4f, Classification Accuracy: %.2f" %(epoch, args.epoch,
                    i+1, total_step, loss_tracker//100, loss.item(), correct/total*100))

def test_cnn(model, test_loader, device, args):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs, classification, mean, var = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

def test_vae(model, test_loader, device, args):
    """# test classification
    model.to('cpu')
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0], data[1]
            outputs, classification, mean, var = model(images)
            _, predicted = torch.max(classification.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    """

    model.to('cpu')
    model.eval()
    # sample images
    with torch.no_grad():
        z = torch.randn(64, args.zdim)
        sample = model.decoder(z)
        save_image(sample.view(64, 1, 28, 28), args.directory)

def show_image(img):
    img = to_img(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def visualise_output(images, model):

    with torch.no_grad():
    
        images = images.to(device)
        images, _, _ = model(images)
        images = images.cpu()
        images = to_img(images)
        np_imagegrid = torchvision.utils.make_grid(images[1:50], 10, 5).numpy()
        plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
        plt.show()

def to_img(x):
    x = x.clamp(0, 1)
    return x


if __name__ == "__main__":
    args = config()
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.cuda) else "cpu")

    transform = transforms.Compose(
        [transforms.ToTensor()]) 
    trainset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    testset = datasets.MNIST('./data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    if args.train:
        if args.model == 'vae':
            model = VAE(args).to(device)
            train_vae(model, train_loader, device, args)
            if args.test:
                #test_vae(model, test_loader, device, args)
                model.train()
                images, labels = iter(test_loader).next()
                images = images.to(device)
                labels = labels.to(device)
                print('Original images')
                show_image(torchvision.utils.make_grid(images[1:50],10,5))
                plt.show()

                # Reconstruct and visualise the images using the vae
                print('VAE reconstruction:')
                visualise_output(images, model)

        elif args.model == 'classifier':
            model = CNN(args).to(device)
            train_vae(model, train_loader, device, args)
            if args.test:
                test_cnn(model, test_loader, device, args)
    