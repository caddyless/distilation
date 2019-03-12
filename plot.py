import csv
import os
import matplotlib as mpb
from init import args
mpb.use('Agg')
import matplotlib.pyplot as plt


def plot_train_trace(train_trace, filename):
    path = 'train_trace/%s-%d-%d-%f/' % (args.method, args.num_worker, args.epoch, args.ratio)
    if os.path.isfile(path + 'record/' + filename + '.csv'):
        print('file existed! rename it..')
        new_name = 'r-' + filename
        while os.path.isfile(path + 'record/' + new_name + '.csv'):
            new_name = 'r-' + new_name
        os.rename(path + 'record/' + filename + '.csv', path + 'record/' + new_name + '.csv')
    with open(path + 'record/' + filename + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('epoch', 'accuracy'))
        for row in train_trace:
            writer.writerow(row)
    epoch, accuracy = zip(*train_trace)
    plt.figure()
    plt.plot(epoch, accuracy, 'r')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    if os.path.isfile(path + 'image/' + filename + '.png'):
        print('file existed! rename it..')
        new_name = 'r-' + filename
        while os.path.isfile(path + 'image/' + new_name + '.png'):
            new_name = 'r-' + new_name
        os.rename(path + 'image/' + filename + '.png', path + 'image/' + new_name + '.png')
    plt.savefig(path + 'image/' + filename + '.png')
