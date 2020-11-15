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
from PIL import Image

log = []
epsilon_log = []
loss_curves = []
acc_curves = []
classification_loss_curves = []
lambdas = [1, 0.5, 0.2, 0.1, 0.01, 0.001, 0.0001, 1e-5, 1e-6, 0]
epsilons = [1, 0.5, 0.3, 0.2, 0.15, 0.1, 0.0007]
#lambdas = [1, 0.1, 0.01, 0.001, 0.0001, 1e-5, 1e-6, 0]
#lambdas = [1, 1e-4, 1e-6, 0]

def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def config():
    parser = argparse.ArgumentParser(description="VAE CLASSIFIER")
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
    parser.add_argument('--save_model', default=False, type=str2bool)
    parser.add_argument("--data", default="MNIST", type=str)
    parser.add_argument("--epsilon", default=0.2, type=float)
    parser.add_argument("--iteration", default=1, type=int)
    parser.add_argument("--tmodel", default="c", type=str)
    parser.add_argument("--vmodel", default="c", type=str)
    parser.add_argument("--attack", default="fgsm", type=str)
    parser.add_argument
    args = parser.parse_args()
    return args


def vae_loss(recon_x, x, mu, logvar):
    # recon_x is the probability of a multivariate Bernoulli distribution p.
    # -log(p(x)) is then the pixel-wise binary cross-entropy.
    # Averaging or not averaging the binary cross-entropy over all pixels here
    # is a subtle detail with big effect on training, since it changes the weight
    # we need to pick for the other loss term by several orders of magnitude.
    # Not averaging is the direct implementation of the negative log likelihood,
    # but averaging makes the weight of the other loss term independent of the image resolution.

    #recon_loss = F.binary_cross_entropy(recon_x.view(-1, 3*32*32), x.view(-1, 3*32*32), reduction='sum')
    recon_loss = F.mse_loss(recon_x.view(-1, args.channels*32*32), x.view(-1, args.channels*32*32), reduction='sum')
    
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

    if args.save_model:
        if input("Save [Y/N]: ").lower() == "y":
            torch.save(model.state_dict(), 'models/vae.pt')
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
    if args.save_model:
        if input("Save [Y/N]: ").lower() == "y":
            torch.save(model.state_dict(), 'models/cnn.pt')


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
            if args.data == "CIFAR":
                save_image(sample.view(64, args.channels, 32, 32), args.directory)
            else:
                save_image(sample.view(64, args.channels, 32, 32), args.directory)


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
                    if args.data== "CIFAR":
                        save_image(sample.view(64, args.channels, 32, 32), "epoch_samples/epoch_sample_%d.png" % epoch)
                    else:
                        save_image(sample.view(64, args.channels, 32, 32), "epoch_samples/epoch_sample_%d.png" % epoch)
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


def train_fgsm(train_loader, test_loader, device, args, num=0, iterations=1):
    if args.tmodel == "v":
        try:
            model = VAE(args)
            model.load_state_dict(torch.load("models/vae.pt"))
        except FileNotFoundError:
            print("VAE model doesn't exist or isn't saved, please try again later")

    elif args.tmodel == "c":
        try:
            model = CNN(args)
            model.load_state_dict(torch.load("models/cnn.pt"))
        except FileNotFoundError:
            print("CNN model doesn't exist or isn't saved, please try again later")

    else:
        print("BAD INPUT: End of Program")
        return -1

    model.eval()

    for i, (image, l) in enumerate(train_loader):
        img = image.to("cpu")
        label = l.to("cpu")

    if args.model != "ifgsm":
        if args.data == "CIFAR":
            save_image(img.view(args.channels, 32, 32), "normal_example.png")
        else: 
            save_image(img.view(args.channels, 32, 32), "normal_example.png")
        img.requires_grad = True
    elif num != 0:
        img = Image.open("adversarial_imgs/adversarial_example_%f.png" % (args.epsilon * iterations))
        img = img.convert('L') 
        img = transform(img).float()
        img = Variable(torch.Tensor(img), requires_grad=True)
        img = img.unsqueeze(0)
        img.retain_grad()
    else:
        img.requires_grad = True

    if args.tmodel == 'v':
        outputs, classification, mean, var = model(img)
        _, loss = t_loss(img, outputs, classification, mean, var, label, args)
        loss.backward()

    elif args.tmodel == 'c':
        classification = model(img)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(classification, label)
        loss.backward()
    
    confidencer = nn.Softmax()
    img = img + img.grad.sign() * args.epsilon
    if args.data == "CIFAR":
        save_image(img.view(args.channels, 32, 32), "adversarial_imgs/adversarial_example_%f.png" % (args.epsilon * iterations))
    else:
        save_image(img.view(args.channels, 32, 32), "adversarial_imgs/adversarial_example_%f.png" % (args.epsilon * iterations))


def test_adversarial_example(args, iterations=1):
    if args.vmodel == 'v':
        try:
            model = VAE(args)
            model.load_state_dict(torch.load("models/vae.pt"))
        except FileNotFoundError:
            print("VAE model doesn't exist or isn't saved, please try again later")

    elif args.vmodel == 'c':
        try:
            model = CNN(args)
            model.load_state_dict(torch.load("models/cnn.pt"))
        except FileNotFoundError:
            print("CNN model doesn't exist or isn't saved, please try again later")

    else:
        print("BAD INPUT: End of Program")
        return -1
    
    transform = transforms.Compose([transforms.Resize([32, 32]),
                                    transforms.ToTensor(),  # gray -> GRB 3 channel (lambda function)
                                    #transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
                                    ])
    
    alt_img = Image.open("adversarial_imgs/adversarial_example_%f.png" % (args.epsilon * iterations))
    alt_img = alt_img.convert('L') 
    alt_img = transform(alt_img).float()
    alt_img = Variable(torch.Tensor(alt_img), requires_grad=True)
    alt_img = alt_img.unsqueeze(0)

    o_img = Image.open("normal_example.png")
    o_img = o_img.convert('L') 
    o_img = transform(o_img).float()
    o_img = Variable(torch.Tensor(o_img), requires_grad=True)
    o_img = o_img.unsqueeze(0)

    model.eval()
    if args.vmodel == 'v':
        alt_outputs, alt_classification, alt_mean, alt_var = model(alt_img)
        o_outputs, o_classification, o_mean, o_var = model(o_img)

    elif args.vmodel == 'c':
        alt_classification = model(alt_img)
        o_classification = model(o_img)

    confidence_fn = nn.Softmax()
    alt_confidence = confidence_fn(alt_classification)
    o_confidence = confidence_fn(o_classification)

    print("\nTarget: %s Tester: %s" %(args.tmodel, args.vmodel))
    print("Epsilon: %f Original Confidences (class: confidence%%)" % (args.epsilon * iterations))
    if args.save_data:
        epsilon_log.append("\nTarget: %s Tester: %s\n" %(args.tmodel, args.vmodel))
        epsilon_log.append("Epsilon: %f Original Confidences (class: confidence%%)\n" %  (args.epsilon * iterations))
    for i in range(o_confidence.shape[1]):
        print(str(i) + ": " + str(o_confidence[0][i].item() * 100) + "%")
        if args.save_data:
            epsilon_log.append(str(i) + ": " + str(o_confidence[0][i].item() * 100) + "%\n")
    print("Epsilon: %f Altered Confidences (class: confidence%%)" % (args.epsilon * iterations))
    if args.save_data:
        epsilon_log.append('\n')
        epsilon_log.append("Epsilon: %f Altered Confidences (class: confidence%%)\n" % (args.epsilon * iterations))
    for i in range(alt_confidence.shape[1]):
        print(str(i) + ": " + str(alt_confidence[0][i].item() * 100) + "%")
        if args.save_data:
            epsilon_log.append(str(i) + ": " + str(alt_confidence[0][i].item() * 100) + "%\n")
    
def loop_thru_epsilon(train_loader, test_loader, device, args):
    for epsilon in epsilons:
        args.epsilon = epsilon
        if args.attack == "fgsm":
            train_fgsm(train_loader, test_loader, device, args)
        elif args.attack == "ifgsm":
            train_ifgsm(train_loader, test_loader, device, args.iteration, args)
        test_adversarial_example(args, args.iteration)
        epsilon_log.append("-" * 20 + "\n")
    
    write_log(epsilon_log, "epsilon-log.txt")
    return

def write_log(data, loc):
    file = open(loc, 'w')
    for line in data:
        file.write(line)
    
    file.close()
    return 

def train_ifgsm(train_loader, test_loader, device, iterations, args):
    args.epsilon = args.epsilon / iterations
    for i in range(iterations):
        train_fgsm(train_loader, test_loader, device, args, num=i, iterations=iterations)

if __name__ == "__main__":
    args = config()
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.cuda) else "cpu")

    #transform = transforms.Compose(
    #    [transforms.ToTensor()])
    if args.model == "gen_ad" or args.model == "test_ad" or args.model == "ad_run" or args.model == "loop_ad" or args.model == "ifgsm":
        args.batch_size = 1
    if args.data == "MNIST":
        args.channels = 1
        transform = transforms.Compose([transforms.Resize([32, 32]),
                                    transforms.ToTensor(),
                                    # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # gray -> GRB 3 channel (lambda function)
                                    # transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
                                    transforms.Normalize((0.5,), (0.5,))
                                    ])  # for grayscale images  
        trainset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        testset = datasets.MNIST('./data', train=False, transform=transform, download=True)
    elif args.data == "CIFAR":
        args.channels = 3
        transform = transforms.Compose(
                            [transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        testset = datasets.CIFAR10('./data', train=True, download=True, transfrom=transform)
    if args.batch_size == 1:
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False)
    else:
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
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

        elif args.model == 'gen_ad':
            train_fgsm(train_loader, test_loader, device, args)
        
        elif args.model == "test_ad":
            test_adversarial_example(args)
        
        elif args.model == "ad_run":
            train_fgsm(train_loader, test_loader, device, args)
            test_adversarial_example(args)
            if args.save_data:
                write_log(epsilon_log, "epsilon-log.txt")
        
        elif args.model == "loop_ad":
            loop_thru_epsilon(train_loader, test_loader, device, args)
        
        elif args.model == "ifgsm":
            train_ifgsm(train_loader, test_loader, device, args.iteration, args)
            test_adversarial_example(args, args.iteration)