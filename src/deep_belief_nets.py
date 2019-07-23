#!/usr/bin/env python3

def load_dataset(filename):
    print('Loading data from ' + filename + ' file...')
    import scipy.io
    mat = scipy.io.loadmat(filename)
    train_data = mat['train_data']
    test_data = mat['test_data']
    train_labels = [item for sublist in mat['train_labels'] for item in sublist]
    test_labels = [item for sublist in mat['test_labels'] for item in sublist]
    return train_data, train_labels, test_data, test_labels

def filter_dataset(train_data, train_labels, test_data, test_labels):
    print('Filtering dataset to 34-66 classes...')
    del_ids = list(range(1, 34, 1)) + list(range(67, 150, 1))
    train_labels, train_data = zip(*((id, train_data) for id, train_data in zip(train_labels, train_data) if id not in del_ids))
    test_labels, test_data = zip(*((id, test_data) for id, test_data in zip(test_labels, test_data) if id not in del_ids))
    return train_data, train_labels, test_data, test_labels

def plot_class_histogram(train_data, test_data, title, filename):
    print('Plotting ' + filename + ' image...')
    import numpy as np
    import random
    from matplotlib import pyplot as plt
    bins = np.arange(-100, 100, 1)
    plt.xlim([min(train_data)-5, max(train_data)+5])
    plt.hist(train_data, bins=bins, alpha=0.5, color='green', label='train dataset')
    plt.hist(test_data, bins=bins, alpha=0.5, color='red', label='test dataset')
    plt.legend()
    plt.title(title)
    plt.xlabel('class')
    plt.ylabel('count')
    plt.savefig(filename)
    plt.close()

if __name__ == '__main__':
    train_data_16, train_labels_16, test_data_16, test_labels_16 = load_dataset('../data/caltech101_silhouettes_16_split1.mat')
    train_data_28, train_labels_28, test_data_28, test_labels_28 = load_dataset('../data/caltech101_silhouettes_28_split1.mat')
    print('Raw train dataset for 16x16 images contains ' + str(len(train_labels_16)) + ' objects')
    print('Raw test dataset for 16x16 images contains ' + str(len(test_labels_16)) + ' objects')
    print('Raw train dataset for 28x28 images contains ' + str(len(train_labels_28)) + ' objects')
    print('Raw test dataset for 28x28 images contains ' + str(len(test_labels_28)) + ' objects')

    train_data_16, train_labels_16, test_data_16, test_labels_16 = filter_dataset(train_data_16, train_labels_16, test_data_16, test_labels_16)
    train_data_28, train_labels_28, test_data_28, test_labels_28 = filter_dataset(train_data_28, train_labels_28, test_data_28, test_labels_28)
    print('Filtered train dataset for 16x16 images contains ' + str(len(train_labels_16)) + ' objects')
    print('Filtered test dataset for 16x16 images contains ' + str(len(test_labels_16)) + ' objects')
    print('Filtered train dataset for 28x28 images contains ' + str(len(train_labels_28)) + ' objects')
    print('Filtered test dataset for 28x28 images contains ' + str(len(test_labels_28)) + ' objects')

#    plot_class_histogram(train_labels_16, test_labels_16, 'Class histogram for 16x16 dataset', 'classes_histogram_for_16_dataset.pdf')
#    plot_class_histogram(train_labels_28, test_labels_28, 'Class histogram for 28x28 dataset', 'classes_histogram_for_28_dataset.pdf')
