import torchvision
import torchvision.transforms as transforms
from torch.utils.data import *
import torch


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