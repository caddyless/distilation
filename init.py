import torch
import argparse
import logging
import os


os.environ["CUDA_VISIABLE_DEVICES"] = '1'
# 从命令行获取参数
parser = argparse.ArgumentParser()
parser.add_argument(
    '-m',
    choices=[
        'batchwise',
        'epochwise',
        'welled',
        'normal'],
    default='welled',
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
parser.add_argument(
    '-n',
    dest='net',
    default='LeNet',
    help='select the net model')
args = parser.parse_args()

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数设置
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

# 检查文件路径
if not os.path.isdir('log'):
    os.mkdir('log')
if not os.path.isdir('log/%s-%d-%d-%f' %
                     (args.method, args.num_worker, args.epoch, args.ratio)):
    os.mkdir('log/%s-%d-%d-%f' %
             (args.method, args.num_worker, args.epoch, args.ratio))
if not os.path.isdir('model'):
    os.mkdir('model')
if not os.path.isdir('model/%s-%d-%d-%f' %
                     (args.method, args.num_worker, args.epoch, args.ratio)):
    os.mkdir('model/%s-%d-%d-%f' %
             (args.method, args.num_worker, args.epoch, args.ratio))
if not os.path.isdir('train_trace'):
    os.mkdir('train_trace')
if not os.path.isdir('train_trace/%s-%d-%d-%f' %
                     (args.method, args.num_worker, args.epoch, args.ratio)):
    os.mkdir('train_trace/%s-%d-%d-%f' %
             (args.method, args.num_worker, args.epoch, args.ratio))
if not os.path.isdir('train_trace/%s-%d-%d-%f/record' %
                     (args.method, args.num_worker, args.epoch, args.ratio)):
    os.mkdir('train_trace/%s-%d-%d-%f/record' %
             (args.method, args.num_worker, args.epoch, args.ratio))
if not os.path.isdir('train_trace/%s-%d-%d-%f/image' %
                     (args.method, args.num_worker, args.epoch, args.ratio)):
    os.mkdir('train_trace/%s-%d-%d-%f/image' %
             (args.method, args.num_worker, args.epoch, args.ratio))

# 设置log
logger_worker = logging.getLogger('worker')
logger_worker.setLevel(level=logging.INFO)
fileHandler = logging.FileHandler(
    'log/%s-%d-%d-%f/worker.log' %
    (args.method, args.num_worker, args.epoch, args.ratio))
fileHandler.setLevel(logging.INFO)
fileHandler.setFormatter(
    logging.Formatter(
        '[%(asctime)s	%(levelname)s]	%(message)s',
        datefmt='%H:%M:%S'))
logger_worker.addHandler(fileHandler)

logger_server = logging.getLogger('server')
logger_server.setLevel(level=logging.INFO)
fileHandler = logging.FileHandler(
    'log/%s-%d-%d-%f/server.log' %
    (args.method, args.num_worker, args.epoch, args.ratio))
fileHandler.setLevel(logging.INFO)
fileHandler.setFormatter(
    logging.Formatter(
        '[%(asctime)s	%(levelname)s]	%(message)s',
        datefmt='%H:%M:%S'))
logger_server.addHandler(fileHandler)

logger_IO = logging.getLogger('IO')
logger_IO.setLevel(level=logging.INFO)
fileHandler = logging.FileHandler(
    'log/%s-%d-%d-%f/IO.log' %
    (args.method, args.num_worker, args.epoch, args.ratio))
fileHandler.setLevel(logging.INFO)
fileHandler.setFormatter(
    logging.Formatter(
        '[%(asctime)s	%(levelname)s]	%(message)s',
        datefmt='%H:%M:%S'))
logger_IO.addHandler(fileHandler)
