import torchvision
import torchvision.transforms as transforms
from torch.utils.data import *
import torch
from init import args
import os
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from init import device
from init import logger_server
import math
import numpy as np
import random


def data_save(dataset, name):
    np_index = []
    np_data = []
    np_label = []
    for sample in dataset:
        index, data, label = sample
        np_index.append(index)
        np_data.append(data)
        np_label.append(label)
    np_index = np.asarray(np_index)
    np_data = np.asarray(np_data)
    np_label = np.asarray(np_label)
    np.savez(name + '.npz', index=np_index, data=np_data, label=np_label)


def data_load(file):
    dataset = np.load(file)
    index = dataset['index'].tolist()
    data = dataset['data'].tolist()
    label = dataset['label'].tolist()
    dataset = zip(index, data, label)
    return dataset


def random_list(length):
    a = []
    for i in range(length):
        b = random.randint(0, 9)
        while b in a:
            b = random.randint(0, 9)
        a.append(b)
    return a


def loss_function(inputs, labels, method='cross_entropy'):
    assert isinstance(inputs, torch.Tensor) and isinstance(
        labels, torch.Tensor), 'input type must be tensor'
    if method == 'cross_entropy':
        dim = inputs.size()
        assert dim == labels.size(), 'size of input and label are not consistent'
        loss = torch.sum(labels.mul(torch.log10(inputs) / math.log10(dim[1])))
        return loss
    elif method == 'distance':
        distance = torch.dist(inputs, labels)
        return distance


def get_data():
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
    )  # 训练数据集
    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform_test)

    print('data collected!')
    return trainset, testset


def data_place(workers, server, beta=3, ratio=0.01, BATCH_SIZE=128):

    print('data placing start...')
    # 获取基本数据
    trainset, testset = get_data()
    num_samples = len(trainset)
    num_workers = len(workers)
    num_class = list(range(0, 49999, 5000))

    # 为每一个训练样本编号  (data, label) -> (id, data, label)
    np_data = [0] * num_samples
    for i in range(num_samples):
        data, label = trainset[i]
        id = num_class[int(label)]
        np_data[id] = data
        num_class[int(label)] += 1
        trainset[i] = (id, data, label)
    np.asarray(np_data, dtype=np.float32)
    np.save('data/mid-data/origin.npy', np_data)

    # 划分私有数据和公有数据
    pub_length = int(num_samples * ratio)
    private_set, public_set = random_split(
        trainset, [num_samples - pub_length, pub_length])
    name = 'data/mid-data/' + str(ratio) + '-' + str(beta)
    data_save(private_set, name + 'pri')
    data_save(public_set, name + 'pub')

    # 根据参数beta划分私有集， beta = n, n in range(0,10), 表示每个worker上只有n种数据
    for worker in workers:
        worker.set_public(public_set)
        selected = random_list(beta)
        private = []
        for sample in private_set:
            if int(sample[2]) in selected:
                if

    print('data placed!')


def train(workers, server, epoch=1, method='batchwise'):
    best_acc = 0.0
    pubset = server.public
    criterion = nn.CrossEntropyLoss()
    if not os.path.isdir('model'):
        os.mkdir('model')
    if not os.path.isdir('image'):
        os.mkdir('image')
    if method == 'batchwise':
        print('train start... method=\'batchwise\' ')
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
                    inputs, ground_truth = sample
                    inputs = Variable(inputs).to(device)
                    optimizer.zero_grad()
                    # forward + backward
                    outputs = server.model(inputs)
                    loss = torch.zeros([1], dtype=torch.float)
                    loss = Variable(loss).to(device)
                    for k, worker in enumerate(workers):
                        labels = worker.model(inputs)
                        # distance = torch.dist(predicted, outputs)
                        loss += loss_function(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    if args.verbose:
                        print('[server:%s epoch:%d batch:%d] Loss: %.03f ' % (
                            server.name, e, j, loss.item()))
                        logger_server.info(
                            '[server:%s epoch:%d batch:%d] Loss: %.03f  ' %
                            (server.name, e, j, loss.item()))
                accuracy = server.evaluation()
                print('epoch finished! Loss: %.03f | Acc: %.3f%% ' %
                      (sum_loss / (i + 1), accuracy))
                if accuracy > best_acc:
                    best_acc = accuracy
                    torch.save(server.model.state_dict(), 'model/tem.pikl')
        if args.plot:
            for worker in workers:
                worker.plot(worker.name)

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
                inputs, ground_truth = sample
                inputs = Variable(inputs).to(device)
                optimizer.zero_grad()
                # forward + backward
                outputs = server.model(inputs)
                loss = torch.zeros(1, dtype=torch.float)
                loss = Variable(loss).to(device)
                for k, worker in enumerate(workers):
                    labels = worker.model(inputs)
                    # distance = torch.dist(predicted, outputs)
                    loss += loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                # 每训练1个batch打印一次loss和准确率
                sum_loss += loss.item()
                if args.verbose:
                    print('[server:%s epoch:%d batch:%d] Loss: %.03f ' % (
                        server.name, e, j, loss.item()))
                    logger_server.info(
                        '[server:%s epoch:%d batch:%d] Loss: %.03f  ' %
                        (server.name, e, j, loss.item()))
            accuracy = server.evaluation()
            print(
                'epoch: %d | Loss: %.03f | Acc: %.3f%% ' %
                (e, sum_loss, accuracy))
            if accuracy > best_acc:
                best_acc = accuracy
                torch.save(server.model.state_dict(), 'model/tem.pikl')
        if args.plot:
            for worker in workers:
                worker.plot(worker.name)

    elif method == 'welled':
        print('train start... method=\'welled\'  ')
        optimizer = optim.Adadelta(
            server.model.parameters(),
            weight_decay=4e-5)
        for worker in workers:
            worker.train(epoch=epoch, method=method)
            if args.plot:
                worker.plot(worker.name)

        for e in range(epoch):
            server.model.train()
            sum_loss = 0.0
            for j, sample in enumerate(pubset):
                # 准备数据
                inputs, ground_truth = sample
                inputs = Variable(inputs).to(device)
                # forward + backward
                outputs = server.model(inputs)
                loss = torch.zeros([1], dtype=torch.float)
                loss = Variable(loss).to(device)
                for k, worker in enumerate(workers):
                    labels = worker.model(inputs)
                    # distance = torch.dist(predicted, outputs)
                    loss += loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                # 每训练1个batch打印一次loss和准确率
                sum_loss += loss.item()
                if args.verbose:
                    print('[server:%s epoch:%d batch:%d] Loss: %.03f ' % (
                        server.name, e, j, loss.item()))
                    logger_server.info(
                        '[server:%s epoch:%d batch:%d] Loss: %.03f  ' %
                        (server.name, e, j, loss.item()))
            accuracy = server.evaluation()
            print(
                'epoch: %d | Loss: %.03f | Acc: %.3f%% ' %
                (e, sum_loss, accuracy))
            if accuracy > best_acc:
                best_acc = accuracy
                torch.save(server.model.state_dict(), 'model/tem.pikl')

    elif method == 'normal':
        print('train start... method=normal. ')
        optimizer = optim.Adadelta(
            server.model.parameters(),
            weight_decay=4e-5)
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
                if args.verbose:
                    print(
                        '[server:%s epoch:%d batch:%d] Loss: %.03f | Acc: %.3f%%' %
                        (server.name, e, j, loss.item(), 100 * correct / total))
                    logger_server.info(
                        '[server:%s epoch:%d batch:%d] Loss: %.03f | Acc: %.3f%%' %
                        (server.name, e, j, loss.item(), 100 * correct / total))
            accuracy = server.evaluation()
            print(
                'epoch: %d | Loss: %.03f | Acc: %.3f%% ' %
                (e, sum_loss, accuracy))
            if accuracy > best_acc:
                best_acc = accuracy
                torch.save(server.model.state_dict(), 'model/tem.pikl')

    print('saving models')
    try:
        os.rename(
            'model/tem.pikl',
            'model/%s-%d-%d-%f/%s.pikl' %
            (args.method,
             args.num_worker,
             args.epoch,
             args.ratio,
             server.name))
    except Exception as e:
        print(e)
        print('rename dir fail\r\n')
    if method != 'normal':
        for worker in workers:
            torch.save(
                worker.model.state_dict(),
                'model/%s-%d-%d-%f/%s.pikl' %
                (args.method,
                 args.num_worker,
                 args.epoch,
                 args.ratio,
                 worker.name))
    print('model saved!')
    if args.plot:
        print('plot...')
        server.plot(server.name)
        print('finished!')
    print('train finished!')
