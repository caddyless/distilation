import matplotlib.pyplot as plt
import csv
import os


def plot_train_trace(train_trace, filename):
    filename = 'image/' + filename
    if os.path.isfile(filename):
        print('file existed! rename it..')
        new_name = 'r-' + filename
        while os.path.isfile(new_name):
            new_name = 'r-' + filename
        os.rename(filename, new_name)
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('epoch', 'accuracy'))
        for row in train_trace:
            writer.writerow(row)
    epoch, accuracy = zip(*train_trace)
    plt.plot(epoch, accuracy, 'r')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
