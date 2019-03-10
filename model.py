from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import *
import torch.nn.functional as F
import argparse
import random
import os
import logging
import resnet


# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 超参数设置
EPOCH = 95  # 遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 128  # 批处理尺寸(batch_size)
LR = 0.001  # 学习率


# Cifar-10的标签
classes = (
    'plane',
    'car',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck')


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


class Server():
    def __init__(self, name):
        self.public = []
        self.test = []
        self.name = name
        self.model = resnet.ResNet18().to(device)

    def set_public(self, public):
        self.public = public

    def set_name(self, name):
        self.name = name

    def set_test(self, test):
        self.test = test

    def evaluation(self):
        print("Waiting Test!")
        self.model.eval()
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


class Worker():
    def __init__(self, name):
        self.private = []
        self.public = []
        self.test = []
        self.name = name
        self.model = resnet.ResNet18().to(device)

    def set_private(self, private):
        self.private = private

    def set_public(self, public):
        self.public = public

    def set_test(self, testloader):
        self.test = testloader

    def train(self, epoch=1, opti='adm', method='batchwise', index=0):
        assert method == 'batchwise' or method == 'epochwise' or method == 'welled', 'method error!'
        # 定义损失函数和优化方式
        criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
        if opti == 'adm':
            # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
            optimizer = optim.Adadelta(
                self.model.parameters(), weight_decay=5e-4)
        elif opti == 'SGD':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=LR,
                momentum=0.9,
                weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

        if method == 'batchwise':
            self.model.train()
            for i, sample in enumerate(self.private, 0):
                if i >= index:
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
                    _, predicted = torch.max(outputs.data, 1)
                    total = labels.size(0)
                    correct = predicted.eq(labels.data).cpu().sum()
                    print('[worker:%s batch:%d] Loss: %.03f | Acc: %.3f%% ' % (
                        self.name, index, loss.item(), 100. * correct / total))
                    logger.info('[worker:%s batch:%d] Loss: %.03f | Acc: %.3f%% ' % (
                        self.name, index, loss.item(), 100. * correct / total))

        elif method == 'epochwise':
            self.model.train()
            sum_loss = 0.0
            correct = 0.0
            total = 0.0
            for i, sample in enumerate(self.private, 0):
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
                sum_loss += loss.item()
                # 每训练1个batch打印一次loss和准确率
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels.data).cpu().sum()
                print('[worker:%s batch:%d] Loss: %.03f | Acc: %.3f%% ' % (
                    self.name, i, loss.item(), 100. * correct / total))
                logger.info('[worker:%s batch:%d] Loss: %.03f | Acc: %.3f%% ' % (
                    self.name, i, loss.item(), 100. * correct / total))
            print('epoch finished! Loss: %.03f | Acc: %.3f%% ' % (sum_loss / i + 1, 100. * correct / total))
            self.evaluation()

        elif method == 'welled':
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
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% ' % (
                        e + 1, (i + 1 + e * length), sum_loss / (i + 1), 100. * correct / total))
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
            accuracy = 100 * correct / total
            print('测试分类准确率为：%.3f%%' % accuracy)
            logger.info('测试分类准确率为：%.3f%%' % accuracy)
        return accuracy

    def predicted(self, sample):
        self.model.eval()
        image, label = sample
        image, label = image.to(device), label.to(device)
        outputs = self.model(image)
        return outputs


def getdata():
    print('collecting data...')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform_train)  # 训练数据集
    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform_test)

    print('data collected!')
    return trainset, testset


def data_place(workers, server, method='iid'):
    assert method == 'iid' or method == 'niid', 'method should be iid or niid!'
    print('data placing start...')
    trainset, testset = getdata()
    length = len(trainset)
    num_workers = len(workers)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2)  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
    server.set_test = testloader
    if method == 'iid':
        datasets = random_split(trainset, int(length / (num_workers + 1)))
        pubset = torch.utils.data.DataLoader(datasets[-1], batch_size=BATCH_SIZE, shuffle=True,
                                                    num_workers=2)
        server.set_public(pubset) # 生成一个个batch进行批训练，组成batch的时候顺序打乱取
        for i, worker in enumerate(workers):
            worker.set_test(testloader)
            private_set = torch.utils.data.DataLoader(datasets[i], batch_size=BATCH_SIZE, shuffle=True,
                                                    num_workers=2)
            worker.set_private(private_set)
            worker.set_public(pubset)

    elif method == 'niid':
        for i, worker in enumerate(workers):
            worker.test = testloader
            weight = list(range(0, 10))
            peek = int((i + 1) / num_workers * 10)
            for j in range(10):
                if j == i:
                    weight[j] = i
                else:
                    weight[j] = abs(1 / (i - j) * (j - 4.5))
            sampler = WeightedRandomSampler(weight, )

    print('data placed!')
    return pubset


def train(workers, server, pubset, epoch=1, method='batchwise'):
    # 定义损失函数和优化方式
    criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
    best_acc = 0.0
    if method == 'batchwise':
        print('train start... method=\'batchwise\'  ')
        length = len(workers[0].private)
        optimizer = optim.Adadelta(server.model.parameters(), weight_decay=4e-5)
        for e in range(epoch):
            for i in range(length):
                for worker in workers:
                    worker.train(method=method, index=i)
                server.model.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for j, sample in enumerate(pubset):
                    # 准备数据
                    inputs, labels = sample
                    inputs, labels = Variable(inputs).to(
                        device), Variable(labels).to(device)
                    optimizer.zero_grad()
                    # forward + backward
                    outputs = server.model(inputs)
                    labels = labels.squeeze_()
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    print('[server:%s batch:%d] Loss: %.03f | Acc: %.3f%% ' % (
                        server.name, j, loss.item(), 100. * correct / total))
                    logger.info('[server:%s batch:%d] Loss: %.03f | Acc: %.3f%% ' % (
                        server.name, j, loss.item(), 100. * correct / total))
                print('epoch finished! Loss: %.03f | Acc: %.3f%% ' % (sum_loss / i + 1, 100. * correct / total))
                accuracy = server.evaluation()
                if accuracy > best_acc:
                    best_acc = accuracy
                    torch.save(server.model.state_dict(), 'tem.pikl')
        print('saving models')
        try:
            os.rename('tem.pikl', '%s-%.3f%%' % (server.name, best_acc))
        except Exception as e:
            print(e)
            print('rename dir fail\r\n')
        for worker in workers:
            torch.save(worker.model.state_dict(), '%s-%.3f%%' % (worker.name, best_acc))
        print('model saved!')

    if method == 'epochwise':
        print('train start... method=\'epochwise\'  ')
        optimizer = optim.Adadelta(server.model.parameters(), weight_decay=4e-5)
        for e in range(epoch):
            for worker in workers:
                worker.train(method=method)
            server.model.train()
            sum_loss = 0.0
            correct = 0.0
            total = 0.0
            for j, sample in enumerate(pubset):
                # 准备数据
                inputs, labels = sample
                inputs, labels = Variable(inputs).to(
                    device), Variable(labels).to(device)
                optimizer.zero_grad()
                # forward + backward
                outputs = server.model(inputs)
                labels = labels.squeeze_()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # 每训练1个batch打印一次loss和准确率
                sum_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels.data).cpu().sum()
                print('[server:%s batch:%d] Loss: %.03f | Acc: %.3f%% ' % (
                    server.name, j, loss.item(), 100. * correct / total))
                logger.info('[server:%s batch:%d] Loss: %.03f | Acc: %.3f%% ' % (
                    server.name, j, loss.item(), 100. * correct / total))
            print('epoch finished! Loss: %.03f | Acc: %.3f%% ' % (sum_loss / i + 1, 100. * correct / total))
            accuracy = server.evaluation()
            if accuracy > best_acc:
                best_acc = accuracy
                torch.save(server.model.state_dict(), 'tem.pikl')
        print('saving models')
        try:
            os.rename('tem.pikl', '%s-%.3f%%' % (server.name, best_acc))
        except Exception as e:
            print(e)
            print('rename dir fail\r\n')
        for worker in workers:
            torch.save(worker.model.state_dict(), '%s-%.3f%%' % (worker.name, best_acc))
        print('model saved!')

    if method == 'welled':
        print('train start... method=\'welled\'  ')
        optimizer = optim.Adadelta(server.model.parameters(), weight_decay=4e-5)
        for worker in workers:
            worker.train(epoch=epoch, method=method)
        for e in range(epoch):
            server.model.train()
            sum_loss = 0.0
            correct = 0.0
            total = 0.0
            for j, sample in enumerate(pubset):
                # 准备数据
                inputs, labels = sample
                inputs, labels = Variable(inputs).to(
                    device), Variable(labels).to(device)
                optimizer.zero_grad()
                # forward + backward
                outputs = server.model(inputs)
                labels = labels.squeeze_()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # 每训练1个batch打印一次loss和准确率
                sum_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels.data).cpu().sum()
                print('[server:%s batch:%d] Loss: %.03f | Acc: %.3f%% ' % (
                    server.name, j, loss.item(), 100. * correct / total))
                logger.info('[server:%s batch:%d] Loss: %.03f | Acc: %.3f%% ' % (
                    server.name, j, loss.item(), 100. * correct / total))
            print('epoch finished! Loss: %.03f | Acc: %.3f%% ' % (sum_loss / i + 1, 100. * correct / total))
            accuracy = server.evaluation()
            if accuracy > best_acc:
                best_acc = accuracy
                torch.save(server.model.state_dict(), 'tem.pikl')
        print('saving models')
        try:
            os.rename('tem.pikl', '%s-%.3f%%' % (server.name, best_acc))
        except Exception as e:
            print(e)
            print('rename dir fail\r\n')
        for worker in workers:
            torch.save(worker.model.state_dict(), '%s-%.3f%%' % (worker.name, best_acc))
        print('model saved!')

    print('train finished!')


if __name__ == '__main__':
    workers = []
    worker1 = Worker('worker1')
    workers.append(worker1)
    worker2 = Worker('worker2')
    workers.append(worker2)
    worker3 = Worker('worker3')
    workers.append(worker3)
    server = Server('server')
    pubset = data_place(workers=workers, server=server)
    train(workers=workers, server=server, pubset=pubset, epoch=135)
