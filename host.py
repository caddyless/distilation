from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
from model import ResNet18
from model import Lenet
from init import logger_worker
from plot import plot_train_trace as ptt
from init import device
from init import args

LR = 0.001  # 学习率


class Host():
    def __init__(self, name):
        self.public = []
        self.test = []
        self.name = name
        self.train_trace = []
        self.verbose = False
        if args.net == 'LeNet':
            self.model = Lenet().to(device)
        elif args.net == 'ResNet18':
            self.model = ResNet18().to(device)

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
            accuracy = 100 * correct / total
            print('测试分类准确率为：%.3f%%' % accuracy)
            logger_worker.info('测试分类准确率为：%.3f%%' % accuracy)
            self.train_trace.append((len(self.train_trace), accuracy))
        return accuracy

    def plot(self, filename):
        ptt(self.train_trace, filename)

    def predicted(self, sample):
        self.model.eval()
        image, label = sample
        image, label = image.to(device), label.to(device)
        outputs = self.model(image)
        return outputs


class Server(Host):
    def __init__(self, name):
        super(Server, self).__init__(name)


class Worker(Host):
    def __init__(self, name):
        super(Worker, self).__init__(name)
        self.private = []
        self.verbose = False

    def set_private(self, private):
        self.private = private

    def data_distri(self):
        distribution = [0] * 10
        for i, sample in enumerate(self.private):
            data, labels = sample
            for l in labels:
                distribution[int(l)] += 1
        return distribution

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
                    if self.verbose:
                        print(
                            '[worker:%s batch:%d] Loss: %.03f | Acc: %.3f%% ' %
                            (self.name, index, loss.item(), 100. * correct / total))
                        logger_worker.info(
                            '[worker:%s batch:%d] Loss: %.03f | Acc: %.3f%% ' %
                            (self.name, index, loss.item(), 100. * correct / total))

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
                if self.verbose:
                    print('[worker:%s batch:%d] Loss: %.03f | Acc: %.3f%% ' % (
                        self.name, i, loss.item(), 100. * correct / total))
                    logger_worker.info(
                        '[worker:%s batch:%d] Loss: %.03f | Acc: %.3f%% ' %
                        (self.name, i, loss.item(), 100. * correct / total))
            print('epoch finished! Loss: %.03f | Acc: %.3f%% ' %
                  (sum_loss / i + 1, 100. * correct / total))

        elif method == 'welled':
            length = len(self.private)
            for e in range(epoch):
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
                    if self.verbose:
                        print('[%s epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% ' % (
                            self.name, e + 1, (i + 1 + e * length), sum_loss / (i + 1), 100. * correct / total))
                        logger_worker.info('%s epoch:%d, iter%d |Loss: %.03f | Acc: %.3f%% ' % (
                            self.name, e + 1, (i + 1 + e * length), sum_loss / (i + 1), 100. * correct / total))
                # 每训练完一个epoch测试一下准确率
                accuracy = self.evaluation()
                print('%s | Epoch : %d | Acc：%.3f%%' % (self.name, e, accuracy))
                logger_worker.info('%s | Epoch : %d | Acc：%.3f%%' % (self.name, e, accuracy))
