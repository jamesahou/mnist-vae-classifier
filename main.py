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
import matplotlib.pyplot as plt
import numpy as np
import time

log = []
loss_curves = []
acc_curves = []
classification_loss_curves = []
#lambdas = [1, 0.5, 0.2, 0.1, 0.01, 0.001, 0.0001, 1e-5, 1e-6, 0]
#lambdas = [1, 0.1, 0.01, 0.001, 0.0001, 1e-5, 1e-6, 0]
lambdas = [1]

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
    parser.add_argument('--lam', default=1.0, type=float)
    parser.add_argument('--eps', default=0, type=float)
    parser.add_argument('--fig_name', default='normal.png', type=str)
    parser.add_argument('--classification', default=True, type=str2bool)
    parser.add_argument('--interval', default=10, type=int)
    parser.add_argument('--save_data', default=True, type=str2bool)
    args = parser.parse_args()
    return args

def show_image(img):
    img = to_img(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def vae_loss(recon_x, x, mu, logvar):
    # recon_x is the probability of a multivariate Bernoulli distribution p.
    # -log(p(x)) is then the pixel-wise binary cross-entropy.
    # Averaging or not averaging the binary cross-entropy over all pixels here
    # is a subtle detail with big effect on training, since it changes the weight
    # we need to pick for the other loss term by several orders of magnitude.
    # Not averaging is the direct implementation of the negative log likelihood,
    # but averaging makes the weight of the other loss term independent of the image resolution.

    recon_loss = F.binary_cross_entropy(recon_x.view(-1, 3*32*32), x.view(-1, 3*32*32), reduction='sum')
    #recon_loss = F.mse_loss(recon_x.view(-1, 3*32*32), x.view(-1, 3*32*32), reduction='sum')
    
    # KL-divergence between the prior distribution over latent vectors
    # (the one we are going to sample from when generating new images)
    # and the distribution estimated by the generator for the given image.
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + 1 * kldivergence

def t_loss(inputs, outputs, c, mean, var, labels, args):
    criterion = nn.CrossEntropyLoss()
    classification_loss = criterion(c, labels)    
    if args.classification:
        c_factor = 1
    else:
        c_factor = 0

    vae_l = vae_loss(outputs, inputs, mean, var)
    return classification_loss * c_factor + args.lam * vae_l, classification_loss

def train_vae(model, train_loader, device, args):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    total_step = len(train_loader)
    acc_list = []
    train_loss_avg = []
    classification_loss = []
    print("Lambda: " + str(args.lam))
    log.append("Lambda: " + str(args.lam) + "\n")
    train_acc_avg = []
    for epoch in range(args.epoch):
        if epoch == 0:
            start_time = time.time()
        loss_tracker = 0
        train_loss_avg.append(0)
        train_acc_avg.append(0)
        classification_loss.append(0)
        batches = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs, classification, mean, var = model(inputs)
            loss, c_loss = t_loss(inputs, outputs, classification, mean, var, labels, args)

            loss.backward()
            optimizer.step()
            
            loss_tracker += loss.item()
            train_loss_avg[-1] += loss.item()
            classification_loss[-1] += c_loss.item()
            total = labels.size(0)
            _, predicted = torch.max(classification.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)
            
            train_acc_avg[-1] += (correct/total)
            
            if (i+1) % 100 == 0:
                #print("Epoch: [%d/%d], Step [%d/%d], Running Loss: %.4f Loss: %.4f, Classification Accuracy: %.2f Reconstruction Loss: %.4f KL Loss: %.4f Class Loss: %.4f" %(epoch, args.epoch,
                #    i+1, total_step, loss_tracker//100, loss.item(), correct/total*100, recon_loss.item(), kl_loss.item(), classification_loss.item()))
                print("Epoch: [%d/%d], Step [%d/%d], Loss: %.4f,  Classification Accuracy: %.2f" % (epoch, args.epoch, i+1, total_step, loss.item(), correct/total*100))
                if args.model == 'run-thru' and args.save_data:
                    file = open("log.txt", 'w')
                    log.append("Epoch: [%d/%d], Step [%d/%d], Loss: %.4f,  Classification Accuracy: %.2f\n" % (epoch, args.epoch, i+1, total_step, loss.item(), correct/total*100))
                    for line in log:
                        file.write(line)
                    file.close()

            batches += 1

        classification_loss[-1] /= batches
        train_loss_avg[-1] /= batches
        train_acc_avg[-1] /= batches

        if epoch == 0:
            total_time = int(time.time() - start_time)
            minutes = total_time // 60
            seconds = total_time - minutes * 60
            print("Time per Epoch %dmin %dsec" %(minutes, seconds))

        if abs(train_loss_avg[-1] - train_loss_avg[len(train_loss_avg)-2])/train_loss_avg[-1] < args.eps:
            break
    
    classification_loss_curves.append(classification_loss)
    acc_curves.append(train_acc_avg)
    loss_curves.append(train_loss_avg)

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
            outputs= model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on the 10000 test images: %.4f %%' % (100 * correct / total))

def test_vae(model, test_loader, device, args):
    # test classification

    model.to('cpu')
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0
    counter = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0], data[1]
            outputs, classification, mean, var = model(images)

            counter += 1
            loss_sum += vae_loss(images, outputs, mean, var).item()

            _, predicted = torch.max(classification.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %.4f %%' % (100 * correct / total))
    print('The average reconstruction loss on the network is %d' % (loss_sum//counter))

    if args.model == 'run-thru' and args.save_data:
        file = open("log.txt", 'w')
        log.append('Accuracy of the network on the 10000 test images: %.4f %%\n' % (100 * correct / total))
        log.append('The average reconstruction loss on the network is %d\n' % (loss_sum//counter))
        for line in log:
            file.write(line)
        file.close()

    model.to('cpu')
    model.eval()
    # sample images
    with torch.no_grad():
        z = torch.randn(64, args.zdim)
        sample = model.decoder(z)
        if args.save_data:
            save_image(sample.view(64, 3, 32, 32), args.directory)

def loop_thru_lambda(trainset, train_loader, testset, test_loader, device, args):
    for i, lam in enumerate(lambdas):
        args.lam = lam
        args.fig_name = str(lam) + "_lambda_model.png"
        args.directory = "samples/" + str(lam) + "-vae.png"
        model = VAE(args).to(device)
        train_vae(model, train_loader, device, args)
        test_vae(model, test_loader, device, args)

    plt.figure()
    for i, lam in enumerate(lambdas):
        plt.plot(acc_curves[i], label="Lambda: %f" % lambdas[i])    
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    if args.save_data:
        plt.savefig("accuracy.png")

    plt.figure()
    for i, lam in enumerate(lambdas):
        plt.plot(loss_curves[i], label="Lambda: %f" % lambdas[i])
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    if args.save_data:
        plt.savefig("loss.png")

    if args.classification:
        plt.figure()
        for i, lam in enumerate(lambdas):
            plt.plot(classification_loss_curves[i], label="Lambda: %f" % lambdas[i])
        plt.title("Classification Loss")
        plt.xlabel("Epoch")
        plt.ylabel("C_Loss")
        plt.legend(loc="upper right")
        if args.save_data:
            plt.savefig("c_loss.jpg")
    
    plt.show()

def loop_epoch(model, train_loader, test_loader, device, args):
    result_acc = []
    result_vae = []
    tested_epochs = []

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    total_step = len(train_loader)
    acc_list = []
    train_loss_avg = []
    classification_loss = []
    print("Lambda: " + str(args.lam))
    log.append("Lambda: " + str(args.lam) + "\n")
    train_acc_avg = []
    for epoch in range(args.epoch):
        if epoch == 0:
            start_time = time.time()
        loss_tracker = 0
        train_loss_avg.append(0)
        train_acc_avg.append(0)
        classification_loss.append(0)
        batches = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs, classification, mean, var = model(inputs)
            loss, c_loss = t_loss(inputs, outputs, classification, mean, var, labels, args)

            loss.backward()
            optimizer.step()
            
            loss_tracker += loss.item()
            train_loss_avg[-1] += loss.item()
            classification_loss[-1] += c_loss.item()
            total = labels.size(0)
            _, predicted = torch.max(classification.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)
            
            train_acc_avg[-1] += (correct/total)
            
            if (i+1) % 100 == 0:
                #print("Epoch: [%d/%d], Step [%d/%d], Running Loss: %.4f Loss: %.4f, Classification Accuracy: %.2f Reconstruction Loss: %.4f KL Loss: %.4f Class Loss: %.4f" %(epoch, args.epoch,
                #    i+1, total_step, loss_tracker//100, loss.item(), correct/total*100, recon_loss.item(), kl_loss.item(), classification_loss.item()))
                print("Epoch: [%d/%d], Step [%d/%d], Loss: %.4f,  Classification Accuracy: %.2f" % (epoch, args.epoch, i+1, total_step, loss.item(), correct/total*100))
                if args.model == 'run-thru' and args.save_data:
                    file = open("log.txt", 'w')
                    log.append("Epoch: [%d/%d], Step [%d/%d], Loss: %.4f,  Classification Accuracy: %.2f\n" % (epoch, args.epoch, i+1, total_step, loss.item(), correct/total*100))
                    for line in log:
                        file.write(line)
                    file.close()

            batches += 1

        classification_loss[-1] /= batches
        train_loss_avg[-1] /= batches
        train_acc_avg[-1] /= batches

        if (epoch+1) % args.interval == 0:
            model.to('cpu')
            model.eval()
            correct = 0
            total = 0
            loss_sum = 0
            counter = 0
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data[0], data[1]
                    outputs, classification, mean, var = model(images)

                    counter += 1
                    loss_sum += vae_loss(images, outputs, mean, var).item()

                    _, predicted = torch.max(classification.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            print('Accuracy of the network on the 10000 test images: %.4f %%' % (100 * correct / total))
            print('The average reconstruction loss on the network is %d' % (loss_sum//counter))

            tested_epochs.append(epoch+1)
            result_acc.append(100 * correct / total)
            result_vae.append(loss_sum//counter)

            if args.model == 'run-thru' and args.save_data:
                file = open("log.txt", 'w')
                log.append('Accuracy of the network on the 10000 test images: %.4f %%\n' % (100 * correct / total))
                log.append('The average reconstruction loss on the network is %d\n' % (loss_sum//counter))
                for line in log:
                    file.write(line)
                file.close()

            model.to('cpu')
            model.eval()
            # sample images
            with torch.no_grad():
                z = torch.randn(64, args.zdim)
                sample = model.decoder(z)
                if args.save_data:
                    save_image(sample.view(64, 3, 32, 32), "epoch_samples/epoch_sample_%d.png" % epoch)
            model.train()
            model.to(device)

        if abs(train_loss_avg[-1] - train_loss_avg[len(train_loss_avg)-2])/train_loss_avg[-1] < args.eps:
            break

        if epoch == 0:
            total_time = int(time.time() - start_time)
            minutes = total_time // 60
            seconds = total_time - minutes * 60
            print("Time per Epoch %dmin %dsec" %(minutes, seconds))
    
    classification_loss_curves.append(classification_loss)
    acc_curves.append(train_acc_avg)
    loss_curves.append(train_loss_avg)
    plt.figure()

    for i, lam in enumerate(lambdas):
        plt.plot(acc_curves[i], label="Lambda: %f" % lambdas[i])    
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    if args.save_data:
        plt.savefig("accuracy.png")

    plt.figure()
    for i, lam in enumerate(lambdas):
        plt.plot(loss_curves[i], label="Lambda: %f" % lambdas[i])
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    if args.save_data:
        plt.savefig("loss.png")

    if args.classification:
        plt.figure()
        for i, lam in enumerate(lambdas):
            plt.plot(classification_loss_curves[i], label="Lambda: %f" % lambdas[i])
        plt.title("Classification Loss")
        plt.xlabel("Epoch")
        plt.ylabel("C_Loss")
        plt.legend(loc="upper right")
        if args.save_data:
            plt.savefig("c_loss.png")
    
    plt.figure()
    plt.plot(tested_epochs, result_acc, label="Test Accuracy over Epochs")
    plt.title("Test Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.xticks(tested_epochs)
    if args.save_data:
        plt.savefig("test_acc_during_training.png")

    plt.figure()
    plt.plot(tested_epochs, result_vae, label="VAE Test Loss over Epochs")
    plt.title("VAE Test Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("VAE Loss")
    plt.xticks(tested_epochs)
    if args.save_data:
        plt.savefig("test_loss_during_training.png")

    plt.show()


if __name__ == "__main__":
    args = config()
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.cuda) else "cpu")

    #transform = transforms.Compose(
    #    [transforms.ToTensor()]) 
    transform = transforms.Compose([transforms.Resize([32, 32]),
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # gray -> GRB 3 channel (lambda function)
                                transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])])  # for grayscale images
    trainset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    testset = datasets.MNIST('./data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    if args.train:
        if args.model == 'vae':
            model = VAE(args).to(device)
            train_vae(model, train_loader, device, args)
            if args.test:
                test_vae(model, test_loader, device, args)
                model.train()

        elif args.model == 'classifier':
            model = CNN(args).to(device)
            train_classifier(model, train_loader, device, args)
            if args.test:
                test_cnn(model, test_loader, device, args)
        
        elif args.model == 'run-thru':
            loop_thru_lambda(trainset, train_loader, testset, test_loader, device, args)

        elif args.model == 'print':
            model = VAE(args)
            model.display()

        elif args.model == 'loop-epoch':
            model = VAE(args).to(device)
            loop_epoch(model, train_loader, test_loader, device, args)
