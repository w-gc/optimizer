# -*- coding: utf-8 -*-
import random
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from resnet import ResNet18
# from MirrorDescent import MirrorDescent
# from MDNesterov import MDNesterov

batch_size = 64
EPOCH = 100
data_dir = 'your_dataset_dir'

# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# dataiter = iter(trainloader)
# inputs, labels = dataiter.next()

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 8, 3)
#         self.conv2 = nn.Conv2d(8, 16, 3)
#         self.conv3 = nn.Conv2d(16, 32, 3)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(32 * 4 * 4, 256)
#         self.fc2 = nn.Linear(256, 64)
#         self.fc3 = nn.Linear(64, 10)
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = F.relu(self.conv3(x))
#         x = x.view(-1, 32 * 4 * 4)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 5, stride = 1, padding = (2, 2))
#         self.maxpool = nn.MaxPool2d(kernel_size = (3, 3), stride = 2, padding=(1, 1))
#         self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 5, stride = 1, padding = (2, 2))
#         self.avgpool = nn.AvgPool2d(kernel_size = (3, 3), stride = 2, padding=(1, 1))
#         self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, stride = 1, padding = (2, 2))
#         self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 4, stride = 1)
#         self.conv5 = nn.Conv2d(in_channels = 64, out_channels = 10, kernel_size = 1, stride = 1)
#     def forward(self, x):
#         batch_size = x.size()[0]
#         x = F.relu(self.maxpool(self.conv1(x)))
#         x = self.avgpool(F.relu(self.conv2(x)))
#         x = self.avgpool(F.relu(self.conv3(x)))
#         x = F.relu(self.conv4(x))
#         x = self.conv5(x)
#         x = x.view(batch_size, -1)
#         return x
        
# net = Net()
net = ResNet18()

use_gpu = torch.cuda.is_available()
if(use_gpu):
    net = net.cuda()

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr = 0.01)
optimizer = optim.SGD(net.parameters(), lr=0.01,  momentum=0.9, weight_decay=5e-4) 
# optimizer = SGD(net.parameters(), lr=1e-2)
# optimizer = MSGD(net.parameters(), lr=0.001, momentum=0.9)
# optimizer = Nesterov(net.parameters(), lr=0.001, momentum=0.9)
# optimizer = AdaGrad(net.parameters())
# optimizer = AdaDelta(net.parameters())
# optimizer = AdaDelta(net.parameters(), lr= 0.001)
# optimizer = RMSProp(net.parameters(), lr= 0.001)
# optimizer = Adam(net.parameters(), lr= 0.001)
# optimizer = Nadam(net.parameters(), lr= 0.001)
# optimizer = ASGD(net.parameters())
# optimizer = SAG(net.parameters())
# optimizer = SVRG(net.parameters(), batch_size = batch_size, epoch = 5)
# optimizer = MirrorDescent(net.parameters(), lr = 0.01, BreDivFun ='Squared norm')
# optimizer = MDNesterov(net.parameters(), lr = 0.01, momentum = 0.8, BreDivFun ='Squared norm')


for epoch in range(EPOCH):  # loop over the dataset multiple times
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        if(use_gpu):
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        # print statistics
        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))

    correct = 0
    total = 0
    with torch.no_grad():
        for (inputs, labels) in testloader:
            if(use_gpu):
                inputs = inputs.cuda()
                labels = labels.cuda()
            outputs = net(inputs)
            outputs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('EPOCH: %d Accuracy of the network on the 10000 test images: %d %%' % (epoch + 1, 100 * correct / total))

# for epoch in range(EPOCH):  # loop over the dataset multiple times
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs
#         inputs, labels = data
#         if(use_gpu):
#             inputs = inputs.cuda()
#             labels = labels.cuda()
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         optimizer.zero_grad()

#         def closure():
#             r = [ random.randint(0, batch_size - 1) ]
#             optimizer.zero_grad()
#             criterion(outputs[r], labels[r]).backward(retain_graph=True)
#             optimizer.save_grad()

#             optimizer.zero_grad()
#             criterion(net(inputs[r]), labels[r]).backward()

#         loss.backward(retain_graph=True)
#         optimizer.step(closure)

#         # print statistics
#         running_loss += loss.item()
#         if i % 100 == 99:    # print every 100 mini-batches
#             print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0

#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data in testloader:
#             inputs, labels = data
#             if(use_gpu):
#                 inputs = inputs.cuda()
#                 labels = labels.cuda()
#             outputs = net(inputs)
#             # outputs = F.softmax(outputs)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     print('%d Accuracy of the network on the 10000 test images: %d %%' % (epoch + 1, 100 * correct / total))

print('---------Finished Training------------')

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for (inputs, labels) in testloader:
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

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))


# # Training
# def train(epoch):
#     print('\nEpoch: %d' % epoch)
#     net.train()
#     train_loss = 0
#     correct = 0
#     total = 0
#     for batch_idx, (inputs, targets) in enumerate(trainloader):
#         inputs, targets = inputs.to(device), targets.to(device)
#         optimizer.zero_grad()
#         outputs = net(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total += targets.size(0)
#         correct += predicted.eq(targets).sum().item()

#         progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                      % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


# def test(epoch):
#     global best_acc
#     net.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(testloader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = net(inputs)
#             loss = criterion(outputs, targets)

#             test_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()

#             progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                          % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

#     # Save checkpoint.
#     acc = 100.*correct/total
#     if acc > best_acc:
#         print('Saving..')
#         state = {
#             'net': net.state_dict(),
#             'acc': acc,
#             'epoch': epoch,
#         }
#         if not os.path.isdir('checkpoint'):
#             os.mkdir('checkpoint')
#         torch.save(state, './checkpoint/ckpt.pth')
#         best_acc = acc


# for epoch in range(start_epoch, start_epoch+200):
#     train(epoch)
#     test(epoch)
