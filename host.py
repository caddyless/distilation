from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
from model import ResNet18
from model import Lenet
from model import autoencoder
from init import logger_worker
from plot import plot_train_trace as ptt
from init import device
from torch.utils.data import *
from torchvision.utils import save_image

LR = 0.001  # 学习率


class Host():
    def __init__(self, name, net):
        self.public = []
        self.test = []
        self.name = name
        self.train_trace = []
        self.verbose = False
        if net == 'LeNet':
            self.model = Lenet().to(device)
        elif net == 'ResNet18':
            self.model = ResNet18().to(device)
        else:
            print('net assigment error!')
            exit()

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

    def statistical(self, set):
        categories = [0] * 10
        if len(set[0]) == 2:
            for sample in set:
                data, label = sample
                categories[int(label)] += 1
        elif len(set[0]) == 3:
            for sample in set:
                index, data, label = sample
                categories[int(label)] += 1
        return categories

    def sta_pub(self):
        return self.statistical(self.public)

    def sta_test(self):
        return self.statistical(self.test)

    def plot(self, filename):
        ptt(self.train_trace, filename)

    def predicted(self, sample):
        self.model.eval()
        image, label = sample
        image, label = image.to(device), label.to(device)
        outputs = self.model(image)
        return outputs


class Server(Host):
    def __init__(self, name, net):
        super(Server, self).__init__(name, net)


class Worker(Host):
    def __init__(self, name, net):
        super(Worker, self).__init__(name, net)
        self.private = []
        self.verbose = False
        self.autoencoder = autoencoder().to(device)

    def sta_private(self):
        return self.statistical(self.private)

    def to_img(self, x):
        x = 0.5 * (x + 1)
        x = x.clamp(0, 1)
        x = x.view(x.size(0), 3, 32, 32)
        return x

    def set_private(self, private):
        self.private = private

    # def train_baseline(self, batch):
    #     private = DataLoader(self.private, batch_size=batch, shuffle=True, num_workers=1)
    #     criterion = nn.CrossEntropyLoss()
    #     optimizer = torch.optim.Adadelta(self.model.parameters(), lr=1e-3, weight_decay=1e-5)

    def train_encoder(self):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adadelta(
            self.autoencoder.parameters(), lr=1e-3, weight_decay=1e-5)
        private = DataLoader(
            self.private,
            batch_size=128,
            shuffle=True,
            num_workers=2)
        for epoch in range(100):
            for data in private:
                index, img, label = data
                img = Variable(img).to(device)
                # ===================forward=====================
                output = self.autoencoder(img)
                loss = criterion(output, img)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # ===================log========================
            print('epoch [{}/{}], loss:{:.4f}'
                  .format(epoch + 1, 100, loss.data[0]))
            if epoch % 10 == 0:
                # pic_gen = self.to_img(output.cpu().data)
                # pic_ori = self.to_img(img)
                save_image(
                    output.cpu().data,
                    './image_{}_gen.png'.format(epoch))
                save_image(img, './image_{}_ori.png'.format(epoch))
        torch.save(
            self.autoencoder.state_dict(),
            './' +
            self.name +
            'autoencoder.pth')

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

        private = DataLoader(
            self.private,
            batch_size=128,
            shuffle=True,
            num_workers=2)
        if method == 'batchwise':
            self.model.train()
            for i, sample in enumerate(private):
                if i >= index:
                    # 准备数据
                    index, inputs, labels = sample
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
            for i, sample in enumerate(private):
                index, inputs, labels = sample
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
                for i, sample in enumerate(private):
                    # 准备数据
                    index, inputs, labels = sample
                    inputs, labels = Variable(inputs).to(
                        device), Variable(labels).to(device)
                    optimizer.zero_grad()

                    # forward + backward
                    outputs = self.model(inputs)
                    labels = labels.squeeze_()
                    loss = criterion(outputs, labels)
                    optimizer.zero_grad()
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
                print(
                    '%s | Epoch : %d | Acc：%.3f%%' %
                    (self.name, e, accuracy))
                logger_worker.info(
                    '%s | Epoch : %d | Acc：%.3f%%' %
                    (self.name, e, accuracy))
