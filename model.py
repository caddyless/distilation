from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import *
from host import Worker
from host import Server
from plot import plot_train_trace as ptt
import torch.nn.functional as F
import argparse
import random
import os
import logging


parser = argparse.ArgumentParser()
parser.add_argument(
    '-m',
    choices=[
        'batchwise',
        'epochwise',
        'welled',
        'normal'],
    default='epochwise',
    dest='method',
    help='the train method selected')
parser.add_argument(
    '-w',
    default='3',
    dest='num_worker',
    type=int,
    help='the number of workers')
parser.add_argument(
    '-r',
    default=0.01,
    dest='ratio',
    type=float,
    help='the ratio between public dataset and the whole dataset')
parser.add_argument(
    '-e',
    default=135,
    dest='epoch',
    type=int,
    help='epoch the server need to be trained')
parser.add_argument(
    '-p',
    action='store_true',
    dest='plot',
    help='plot the train curve or not')
args = parser.parse_args()


# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 超参数设置
EPOCH = 135  # 遍历数据集次数
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


# class DistanceLoss(nn.Module):
#     def __init__(self):
#         super(DistanceLoss, self).__init__()
#         return
#
#     def forward(self, inputs, outputs, workers):
#         for worker in workers:


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


def data_place(workers, server, method='iid', ratio=0.01):
    assert method == 'iid' or method == 'niid', 'method should be iid or niid!'
    print('data placing start...')
    trainset, testset = getdata()
    num_samples = len(trainset)
    num_workers = len(workers)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
    server.set_test(testloader)
    if method == 'iid':
        lengths = []
        worker_samples = int(num_samples * (1 - ratio) / num_workers)
        server_samples = int(num_samples * ratio)
        for w in range(num_workers + 1):
            if w == num_workers - 1:
                lengths.append(
                    num_samples -
                    w *
                    worker_samples -
                    server_samples)
            elif w == num_workers:
                lengths.append(server_samples)
            else:
                lengths.append(worker_samples)
        datasets = random_split(trainset, lengths)
        pubset = torch.utils.data.DataLoader(
            datasets[-1], batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        server.set_public(pubset)  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取
        for i, worker in enumerate(workers):
            worker.set_test(testloader)
            private_set = torch.utils.data.DataLoader(
                datasets[i], batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
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
    best_acc = 0.0
    train_trace = []
    if method == 'batchwise':
        print('train start... method=\'batchwise\'  ')
        length = len(workers[0].private)
        optimizer = optim.Adadelta(
            server.model.parameters(),
            weight_decay=4e-5)
        for e in range(epoch):
            for i in range(length):
                for worker in workers:
                    worker.train(method=method, index=i)
                server.model.train()
                sum_loss = 0.0
                for j, sample in enumerate(pubset):
                    # 准备数据
                    inputs, labels = sample
                    inputs, labels = Variable(inputs).to(
                        device), Variable(labels).to(device)
                    optimizer.zero_grad()
                    # forward + backward
                    outputs = server.model(inputs)
                    loss = torch.zeros([1], dtype=torch.float)
                    loss = Variable(loss).to(device)
                    for k, worker in enumerate(workers):
                        predicted = worker.model(inputs)
                        distance = torch.dist(predicted, outputs)
                        loss += distance
                    loss.backward()
                    optimizer.step()
                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    print('[server:%s epoch:%d batch:%d] Loss: %.03f ' % (
                        server.name, e, j, loss.item()))
                    logger.info(
                        '[server:%s epoch:%d batch:%d] Loss: %.03f  ' %
                        (server.name, e, j, loss.item()))
                accuracy = server.evaluation()
                train_trace.append((len(train_trace), accuracy))
                print('epoch finished! Loss: %.03f | Acc: %.3f%% ' %
                      (sum_loss / (i + 1), accuracy))
                if accuracy > best_acc:
                    best_acc = accuracy
                    torch.save(server.model.state_dict(), 'tem.pikl')

    elif method == 'epochwise':
        print('train start... method=\'epochwise\'  ')
        optimizer = optim.Adadelta(
            server.model.parameters(),
            weight_decay=4e-5)
        for e in range(epoch):
            for worker in workers:
                worker.train(method=method)
            server.model.train()
            sum_loss = 0.0
            for j, sample in enumerate(pubset):
                # 准备数据
                inputs, labels = sample
                inputs, labels = Variable(inputs).to(
                    device), Variable(labels).to(device)
                optimizer.zero_grad()
                # forward + backward
                outputs = server.model(inputs)
                loss = torch.zeros(1, dtype=torch.float)
                loss = Variable(loss).to(device)
                for k, worker in enumerate(workers):
                    predicted = worker.model(inputs)
                    distance = torch.dist(predicted, outputs)
                    loss += distance
                loss.backward()
                optimizer.step()
                # 每训练1个batch打印一次loss和准确率
                sum_loss += loss.item()
                print('[server:%s batch:%d] Loss: %.03f ' % (
                    server.name, j, loss.item()))
                logger.info('[server:%s batch:%d] Loss: %.03f  ' % (
                    server.name, j, loss.item()))
            accuracy = server.evaluation()
            train_trace.append((len(train_trace), accuracy))
            print(
                'epoch finished! Loss: %.03f | Acc: %.3f%% ' %
                (sum_loss, accuracy))
            if accuracy > best_acc:
                best_acc = accuracy
                torch.save(server.model.state_dict(), 'tem.pikl')

    elif method == 'welled':
        print('train start... method=\'welled\'  ')
        optimizer = optim.Adadelta(
            server.model.parameters(),
            weight_decay=4e-5)
        for worker in workers:
            worker.train(epoch=epoch, method=method)
        for e in range(epoch):
            server.model.train()
            sum_loss = 0.0
            for j, sample in enumerate(pubset):
                # 准备数据
                inputs, labels = sample
                inputs, labels = Variable(inputs).to(
                    device), Variable(labels).to(device)
                optimizer.zero_grad()
                # forward + backward
                outputs = server.model(inputs)
                loss = torch.zeros([1], dtype=torch.float)
                loss = Variable(loss).to(device)
                for k, worker in enumerate(workers):
                    predicted = worker.model(inputs)
                    distance = torch.dist(predicted, outputs)
                    loss += distance
                loss.backward()
                optimizer.step()
                # 每训练1个batch打印一次loss和准确率
                sum_loss += loss.item()
                print('[server:%s batch:%d] Loss: %.03f ' % (
                    server.name, j, loss.item()))
                logger.info('[server:%s batch:%d] Loss: %.03f  ' % (
                    server.name, j, loss.item()))
            accuracy = server.evaluation()
            train_trace.append((len(train_trace), accuracy))
            print(
                'epoch finished! Loss: %.03f | Acc: %.3f%% ' %
                (sum_loss, accuracy))
            if accuracy > best_acc:
                best_acc = accuracy
                torch.save(server.model.state_dict(), 'tem.pikl')

    elif method == 'normal':
        print('train start... method=normal. ')
        optimizer = optim.Adadelta(
            server.model.parameters(),
            weight_decay=4e-5)
        criterion = nn.CrossEntropyLoss()
        for e in range(epoch):
            server.model.train()
            sum_loss = 0.0
            total = 0.0
            correct = 0.0
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
                correct += predicted.eq(labels).cpu().sum()
                print(
                    '[server:%s epoch:%d batch:%d] Loss: %.03f | Acc: %.3f%%' %
                    (server.name, e, j, loss.item(), 100 * correct / total))
                logger.info(
                    '[server:%s epoch:%d batch:%d] Loss: %.03f | Acc: %.3f%%' %
                    (server.name, e, j, loss.item(), 100 * correct / total))
            accuracy = server.evaluation()
            train_trace.append((len(train_trace), accuracy))
            print(
                'epoch finished! Loss: %.03f | Acc: %.3f%% ' %
                (sum_loss, accuracy))
            if accuracy > best_acc:
                best_acc = accuracy
                torch.save(server.model.state_dict(), 'tem.pikl')

    print('saving models')
    try:
        os.rename('tem.pikl', '%s-%s-%d-%.2f-%.3f%%.pikl' %
                  (server.name, method, args.num_worker, args.ratio, best_acc))
    except Exception as e:
        print(e)
        print('rename dir fail\r\n')
    for worker in workers:
        torch.save(worker.model.state_dict(), '%s-%s-%d-%.2f-%.3f%%.pikl' %
                   (worker.name, method, args.num_worker, args.ratio, best_acc))
    print('model saved!')
    if args.plot:
        print('plot...')
        ptt(train_trace, '%s-%s-%d-%.2f-%.3f%%.png' %
                   (server.name, method, args.num_worker, args.ratio, best_acc))
        for worker in workers:

        print('finished!')
    print('train finished!')


if __name__ == '__main__':
    workers = []
    for i in range(args.num_worker):
        workers.append(Worker('worker%d' % i))
    server = Server('server')
    pubset = data_place(workers=workers, server=server, ratio=args.ratio)
    train(
        workers=workers,
        server=server,
        pubset=pubset,
        epoch=args.epoch,
        method=args.method)
