# example of loading the cifar10 dataset
import pickle

import numpy as np
from skimage import color


def preprocess():
    '''

    :return:
        X_train: Black-and-white images, in L*a*b space. Dimensions of (num_examples, 32, 32, 1)
        X_test: Black-and-white images, in L*a*b space. Dimensions of (num_labels, 32, 32, 1)
        Y_train: Colored images, in L*a*b space. Dimensions of (num_examples, 32, 32, 3)
        Y_test: Colored images, in L*a*b space. Dimensions of (num_labels, 32, 32, 3)
    '''
    # Load data
    Y_train, Y_test = load_cifar_10_data()
    # Normalize
    Y_train, Y_test = normalize(Y_train), normalize(Y_test)
    # Convert from RGB space to L*a*b color space (https://en.wikipedia.org/wiki/CIELAB_color_space)
    Y_train, Y_test = convertFromRgbToLab(Y_train), convertFromRgbToLab(Y_test)
    # Remove (a, b) channels to create black-and-white images. These will be our domain with colored images as range.
    X_train, X_test = np.reshape(np.copy(Y_train)[:, :, :, 0], [len(Y_train), 32, 32, 1]), \
                      np.reshape(np.copy(Y_test)[:, :, :, 0], [len(Y_test), 32, 32, 1])
    return X_train, X_test, Y_train, Y_test


def convertFromRgbToLab(color_images):
    return np.array([color.rgb2lab(color_images[i]) for i in range(len(color_images))])


def normalize(X):
    # convert from integers to floats
    train_norm = X.astype('float32')
    # normalize to range 0-1
    X_normalized = train_norm / 255.0
    return X_normalized


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_cifar_10_data(data_dir="../cifar-10-batches-py"):
    """
    Loads the train and test images from the CIFAR-10 dataset.
    Assumes you have downloaded the CIFAR-10 zip, unzipped, and are passing the path of the unzipped dir as 'data_dir'
    """
    # training data
    cifar_train_data = None

    # cifar_train_data_dict
    # 'batch_label': 'training batch 5 of 5'
    # 'data': ndarray
    # 'filenames': list
    # 'labels': list

    for i in range(1, 6):
        cifar_train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            cifar_train_data = cifar_train_data_dict[b'data']
        else:
            cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))

    cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
    cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)

    # test data
    # cifar_test_data_dict
    # 'batch_label': 'testing batch 1 of 1'
    # 'data': ndarray
    # 'filenames': list
    # 'labels': list

    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    cifar_test_data = cifar_test_data_dict[b'data']

    cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)

    return cifar_train_data, cifar_test_data


if __name__ == "__main__":
    pass
