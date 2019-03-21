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


def loss_function(inputs, labels, method='cross_entropy'):
    assert isinstance(inputs, torch.Tensor) and isinstance(labels, torch.Tensor),'input type must be tensor'
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
        transform=transform_train)  # 训练数据集
    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform_test)

    print('data collected!')
    return trainset, testset


def data_place(workers, server, method='iid', ratio=0.01, BATCH_SIZE=128):
    assert method == 'iid' or method == 'niid', 'method should be iid or niid!'
    print('data placing start...')
    trainset, testset = get_data()
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


def train(workers, server, epoch=1, method='batchwise'):
    best_acc = 0.0
    pubset = server.public
    criterion = nn.CrossEntropyLoss()
    if not os.path.isdir('model'):
        os.mkdir('model')
    if not os.path.isdir('image'):
        os.mkdir('image')
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
