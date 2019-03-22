from host import Worker
from host import Server
from utils import data_place as dp
from utils import train
from init import args


if __name__ == '__main__':
    # 初始化worker和server
    workers = []
    for i in range(args.num_worker):
        worker = Worker('worker%d' % i)
        worker.verbose = args.verbose
        workers.append(worker)
    server = Server('server')

    # 在worker和server上分布数据
    dp(workers=workers, server=server, ratio=args.ratio)

    # 开始训练
    train(
        workers=workers,
        server=server,
        epoch=args.epoch,
        method=args.method)
