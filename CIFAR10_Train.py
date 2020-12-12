# -*- coding: utf-8 -*-
import random
import csv
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from resnet import *

batch_size = 100
EPOCH = 20
data_dir = 'your_dataset_dir'


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = ResNet34() # ResNet18()

use_gpu = torch.cuda.is_available()
if(use_gpu):
    net = net.cuda()

criterion = nn.CrossEntropyLoss()

# optimizer = optim.SGD(net.parameters(), lr = 1e-2)

def save_result(filename, data):
    with open(filename, "w", newline='') as file:
        writer = csv.writer(file)
        for temp in data:
            writer.writerow([temp])
        file.close()

def train(EPOCH, optimizer, filename):
    print('--------------'+ filename[7:] +'--------------')
    loss_save = []
    epoch_loss_save = []
    epoch_acc_save = []
    for epoch in range(EPOCH): 
        for i, (inputs, labels) in enumerate(trainloader, 0):
            if(use_gpu):
                inputs = inputs.cuda()
                labels = labels.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            loss_save.append(loss.item())
            # print statistics
            if i % 100 == 99:    # print every 100 mini-batches
                print('[%d, %d] loss: %.5f' % (epoch + 1, i + 1, loss.item()))

        correct = 0
        total = 0
        with torch.no_grad():
            for (inputs, labels) in testloader:
                if(use_gpu):
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                epoch_loss_save.append(loss.item())
                epoch_acc_save.append(correct / total)

        print('EPOCH: %d Accuracy of the network on the 10000 test images: %d %%' % (epoch + 1, 100 * correct / total))
    
    save_result(filename = filename + '_loss.csv', data = loss_save)
    save_result(filename = filename + '_epoch_loss.csv', data = epoch_loss_save)
    save_result(filename = filename + '_epoch_acc.csv', data = epoch_acc_save)
    
    print('---------Finished Training------------')
    
def test(filename):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            if(use_gpu):
                inputs = inputs.cuda()
                labels = labels.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    class_acc = list(range(10))
    for i in range(10):
        class_acc[i] = class_correct[i] / class_total[i]
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_acc[i]))

    save_result(filename = filename + '_test_results.csv', data = class_acc)


def comparison(EPOCH=EPOCH, batch_size = batch_size):
    def sgd():
        from SGD import SGD
        return SGD(net.parameters(), lr=0.001)
    
    def msgd():
        from MSGD import MSGD
        return MSGD(net.parameters(), lr=0.001, momentum=0.9)
    
    def nesterov():
        from Nesterov import Nesterov
        return Nesterov(net.parameters(), lr=0.001, momentum=0.9)
    
    def ada_grad():
        from AdaGrad import AdaGrad
        return AdaGrad(net.parameters())
    
    def ada_delta():
        from AdaDelta import AdaDelta
        return AdaDelta(net.parameters())
    
    def rms_prop():
        from RMSProp import RMSProp
        return RMSProp(net.parameters(), lr= 0.001)
    
    def adam():
        from Adam import Adam
        return Adam(net.parameters(), lr= 0.001)
    
    def n_adam():
        from Nadam import Nadam
        return Nadam(net.parameters(), lr= 0.001)
    
    def asgd():
        from ASGD import ASGD
        return ASGD(net.parameters())
    
    def sag():
        from SAG import SAG
        return SAG(net.parameters())
    
    def svrg():
        from SVRG import SVRG
        return SVRG(net.parameters(), batch_size = batch_size, epoch = 5)
    
    def mirror_descent():
        from MirrorDescent import MirrorDescent
        return MirrorDescent(net.parameters(), lr = 0.01, BreDivFun ='Squared norm')
    
    def md_nesterov():
        from MDNesterov import MDNesterov
        return MDNesterov(net.parameters(), lr = 0.01, momentum = 0.8, BreDivFun ='Squared norm')

    optimizers = {'SGD': sgd,
            'MSGD': msgd, 
            'Nesterov': nesterov,
            'AdaGrad': ada_grad,
            'AdaDelta': ada_delta,
            'RMSProp': rms_prop,
            'Adam': adam,
            'Nadam': n_adam,
            'ASGD': asgd,
            'SAG': sag,
            # 'SVRG': svrg,
            'MirrorDescent': mirror_descent,
            'MDNesterov': md_nesterov}
    for key in optimizers:
        train(EPOCH=EPOCH, optimizer = optimizers[key](), filename='result_resnet34/'+ key)
        test(filename='result_resnet34/'+ key)

comparison(EPOCH=EPOCH, batch_size = batch_size)


# train(EPOCH=EPOCH, optimizer = optim.SGD(net.parameters(), lr = 1e-2), filename='SGD')
# test(filename='SGD')
# python CIFAR10_Train.py
