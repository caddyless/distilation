from host import Worker
from host import Server
from utils import data_place as dp
from utils import train
from utils import train_baseline
from init import args


if __name__ == '__main__':
    # 初始化worker和server
    workers = []
    for i in range(args.num_worker):
        worker = Worker('worker%d' % i, 'LeNet')
        worker.verbose = args.verbose
        workers.append(worker)
    server = Server('server', 'LeNet')

    # server.train_normal()

    # 在worker和server上分布数据
    dp(workers=workers, server=server, ratio=args.ratio)
    train_baseline(workers=workers, server=server)
    # workers[0].train_encoder()
    # 开始训练
    # train(
    #     workers=workers,
    #     server=server,
    #     epoch=args.epoch,
    #     method=args.method)
