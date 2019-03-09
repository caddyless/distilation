from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import csv
import argparse
import random
import os
import logging
import resnet


# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 超参数设置
EPOCH = 95   #遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 128      #批处理尺寸(batch_size)
LR = 0.1        #学习率


# Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 设置log
fileHandler = logging.FileHandler('cifar10-net.log')
fileHandler.setLevel(logging.INFO)
fileHandler.setFormatter(
    logging.Formatter(
        '[%(asctime)s	%(levelname)s]	%(message)s',
        datefmt='%H:%M:%S'))
logging.getLogger("cifar10-net").setLevel(logging.WARNING)
logging.basicConfig(level=logging.DEBUG, handlers=[fileHandler])
logger = logging.getLogger("cifar10-net")


class Worker():
    def __init__(self):
        self.private = []
        self.public = []
        self.test = []
        self.model = resnet.ResNet18().to(device)

    def get_private(self, private):
        self.private = private

    def get_public(self, public):
        self.public = public

    def train(self, epoch, method = 'adm'):
        # 定义损失函数和优化方式
        criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
        optimizer = optim.Adadelta(self.model.parameters(), weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
        length = len(self.private)
        for e in epoch:
            self.model.train()
            sum_loss = 0.0
            correct = 0.0
            total = 0.0
            for i, sample in enumerate(self.private, 0):
                # 准备数据
                inputs, labels = sample
                inputs, labels = Variable(inputs).to(
                    device), Variable(labels).to(device)
                optimizer.zero_grad()

                # forward + backward
                outputs = self.model(inputs)
                labels = labels.squeeze_()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # 每训练1个batch打印一次loss和准确率
                sum_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels.data).cpu().sum()
                print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                      % (e + 1, (i + 1 + e * length), sum_loss / (i + 1), 100. * correct / total))
                logger.info('%03d  %05d |Loss: %.03f | Acc: %.3f%% ' % (
                    e + 1, (i + 1 + e * length), sum_loss / (i + 1), 100. * correct / total))
            # 每训练完一个epoch测试一下准确率
            self.evaluation()

    def evaluation(self):
        print("Waiting Test!")
        with torch.no_grad():
            correct = 0
            total = 0
            for sample in self.test:
                self.model.eval()
                image, label = sample
                image, label = image.to(device), label.to(device)
                outputs = self.model(image)
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                label = label.squeeze_()
                correct += (predicted == label).sum().item()
            print('测试分类准确率为：%.3f%%' % (100 * correct / total))
            logger.info('测试分类准确率为：%.3f%%' % (100 * correct / total))


def getdata():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                            transform=transform_train)  # 训练数据集
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    return trainset, testset


def data_split(workers, server):
    trainset, testset = getdata()
    num_workers = len(workers)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                                              num_workers=2)  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)



def train(trainloader,testloader, net):
    net.to(device)
    # 定义损失函数和优化方式
    criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
    optimizer = optim.Adadelta(net.parameters(), weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    best_acc = 0.85
    length = len(trainloader)
    for epoch in range(pre_epoch, EPOCH):
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, sample in enumerate(trainloader, 0):
            # 准备数据
            if i >= 0.6*length:
                inputs, labels = sample
                inputs, labels = Variable(inputs).to(
                    device), Variable(labels).to(device)
                optimizer.zero_grad()

                # forward + backward
                outputs = net(inputs)
                labels = labels.squeeze_()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # 每训练1个batch打印一次loss和准确率
                sum_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels.data).cpu().sum()
                print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                      % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                logger.info('%03d  %05d |Loss: %.03f | Acc: %.3f%% ' % (
                    epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
            else:
                continue
        # 每训练完一个epoch测试一下准确率
        print("Waiting Test!")
        with torch.no_grad():
            correct = 0
            total = 0
            for sample in testloader:
                net.eval()
                image, label = sample
                image, label = image.to(device), label.to(device)
                outputs = net(image)
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                label = label.squeeze_()
                correct += (predicted == label).sum().item()
            print('测试分类准确率为：%.3f%%' % (100 * correct / total))
            logger.info('测试分类准确率为：%.3f%%' % (100 * correct / total))
            acc = 100 * correct / total
            if acc > best_acc:
                best_acc = acc
    print('train finished!')


if __name__ == '__main__':
    trainloader, testloader = getdata()
    server = resnet.ResNet18().to(device)
    worker1 = Worker()
    worker2 = Worker()
    train(trainloader, testloader, net)
