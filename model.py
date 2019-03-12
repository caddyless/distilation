from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
from host import Worker
from host import Server
from data import data_place as dp
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
parser.add_argument(
    '-v',
    action='store_true',
    dest='verbose',
    help='show the detail or not')
args = parser.parse_args()


# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 超参数设置
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
fileHandler = logging.FileHandler('model.log')
fileHandler.setLevel(logging.INFO)
fileHandler.setFormatter(
    logging.Formatter(
        '[%(asctime)s	%(levelname)s]	%(message)s',
        datefmt='%H:%M:%S'))
logging.getLogger("model").setLevel(logging.WARNING)
logging.basicConfig(level=logging.DEBUG, handlers=[fileHandler])
logger = logging.getLogger("model")


def train(workers, server, epoch=1, method='batchwise'):
    best_acc = 0.0
    pubset = server.public
    if not os.path.isdir('model'):
        os.mkdir('model')
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
                    if args.verbose:
                        print('[server:%s epoch:%d batch:%d] Loss: %.03f ' % (
                            server.name, e, j, loss.item()))
                        logger.info(
                            '[server:%s epoch:%d batch:%d] Loss: %.03f  ' %
                            (server.name, e, j, loss.item()))
                accuracy = server.evaluation()
                print('epoch finished! Loss: %.03f | Acc: %.3f%% ' %
                      (sum_loss / (i + 1), accuracy))
                if accuracy > best_acc:
                    best_acc = accuracy
                    torch.save(server.model.state_dict(), 'model/tem.pikl')

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
                if args.verbose:
                    print('[server:%s epoch:%d batch:%d] Loss: %.03f ' % (
                        server.name, e, j, loss.item()))
                    logger.info(
                        '[server:%s epoch:%d batch:%d] Loss: %.03f  ' %
                        (server.name, e, j, loss.item()))
            accuracy = server.evaluation()
            print(
                'epoch: %d | Loss: %.03f | Acc: %.3f%% ' %
                (e, sum_loss, accuracy))
            if accuracy > best_acc:
                best_acc = accuracy
                torch.save(server.model.state_dict(), 'model/tem.pikl')

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
                if args.verbose:
                    print('[server:%s epoch:%d batch:%d] Loss: %.03f ' % (
                        server.name, e, j, loss.item()))
                    logger.info(
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
                if args.verbose:
                    print(
                        '[server:%s epoch:%d batch:%d] Loss: %.03f | Acc: %.3f%%' %
                        (server.name, e, j, loss.item(), 100 * correct / total))
                    logger.info(
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
        os.rename('model/tem.pikl', 'model/%s-%s-%d-%.2f-%.3f%%.pikl' %
                  (server.name, method, args.num_worker, args.ratio, best_acc))
    except Exception as e:
        print(e)
        print('rename dir fail\r\n')
    if method != 'normal':
        for worker in workers:
            torch.save(worker.model.state_dict(), 'model/%s-%s-%d-%.2f-%.3f%%.pikl' %
                       (worker.name, method, args.num_worker, args.ratio, best_acc))
    print('model saved!')
    if args.plot:
        print('plot...')
        server.plot('%s-%s-%d-%.2f-%.3f%%.png' %
                   (server.name, method, args.num_worker, args.ratio, best_acc))
        for worker in workers:
            worker.plot('%s-%s-%d-%.2f-%.3f%%.png' %
                   (worker.name, method, args.num_worker, args.ratio, best_acc))

        print('finished!')
    print('train finished!')


if __name__ == '__main__':
    workers = []
    for i in range(args.num_worker):
        worker = Worker('worker%d' % i)
        worker.verbose = args.verbose
        workers.append(worker)
    server = Server('server')
    pubset = dp(workers=workers, server=server, ratio=args.ratio)
    train(
        workers=workers,
        server=server,
        epoch=args.epoch,
        method=args.method)
