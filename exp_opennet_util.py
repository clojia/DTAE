import datetime
import gzip
import os.path
#import cPickle
import pickle
import sys
import time
import keras
import numpy as np
import scipy.io
import transformations as ts
import stl10_input
from sklearn import preprocessing
from tensorflow.examples.tutorials.mnist import input_data
from keras.datasets import cifar10, cifar100
import tensorflow as tf
import cv2
import random
from datetime import datetime

sys.path.insert(0, os.path.abspath("./simple-dnn"))

from simple_dnn.util.format import UnitPosNegScale, reshape_pad
from util.openworld_sim import OpenWorldSim, OpenWorldMsData

def save_pickle_gz(obj, filename, protocol=1):
    """Saves a compressed object to disk
    """
    with gzip.GzipFile(filename, 'wb') as fout:
        pickle.dump(obj, fout, protocol)

def load_pickle_gz(filename):
    """Loads a compressed object from disk
    """
    with gzip.GzipFile(filename, 'rb') as fin:
        obj = pickle.load(fin)

    return obj


def one_hot_convert(vec):
    one_hot_vec = np.zeros((vec.size, vec.max() + 1))
    one_hot_vec[np.arange(vec.size), vec] = 1
    return one_hot_vec


def load_trans_data(x_train, y_train, n_rots):
    n_train, n_dims = x_train.shape
    ws = np.random.randn(n_rots, n_dims, n_dims)
    # bs_train = np.random.randn(n_rots, n_train, n_dims)
    train_rot = [0] * n_train
    x_train_trans = x_train
    y_train_trans = y_train
    for i, w in zip(range(1, n_rots + 1), ws):
        x_train_trans = np.concatenate([x_train_trans, x_train.dot(w)])
        train_rot = np.concatenate([train_rot, [i] * n_train])
        y_train_trans = np.concatenate([y_train_trans, y_train])
    train_rot_one_hot = one_hot_convert(train_rot)
    return x_train_trans, y_train_trans, train_rot_one_hot

# def random_order_transformation_1d(x_train, y_train, x_dim):
#     n_train, n_dims = x_train.shape
#     #  ws = np.random.randn(n_rots, n_dims, n_dims)
#     # bs_train = np.random.randn(n_rots, n_train, n_dims)
#     train_rot = [0] * n_train
#     x_train_trans = x_train
#     y_train_trans = y_train
#     r = list(range(x_dim))
#     random.seed(datetime.now())
#     for i in range(1):
#    #     rand_r = random.sample(r, len(r))
#     #    print(rand_r)
#         for x in x_train:
#             x_trans = x[::-1].reshape(1, x_dim)
#             x_train_trans = np.concatenate([x_train_trans, x_trans])
#         train_rot = np.concatenate([train_rot, [i + 1] * n_train])
#         y_train_trans = np.concatenate([y_train_trans, y_train])
#     train_rot_one_hot = one_hot_convert(train_rot)
#     return x_train_trans, y_train_trans, train_rot_one_hot


def random_order_transformation(x_train, y_train, x_dim, n_trans):
    n_train, n_dims = x_train.shape
  #  ws = np.random.randn(n_rots, n_dims, n_dims)
   # bs_train = np.random.randn(n_rots, n_train, n_dims)
    train_rot = [0] * n_train
    x_train_trans = x_train
    y_train_trans = y_train
    r = list(range(x_dim))
    random.seed(datetime.now())
    for i in range(n_trans):
        rand_r = random.sample(r, len(r))
        print(rand_r)
        for x in x_train:
            x_trans = x.reshape(x_dim,x_dim)[np.ix_(rand_r, rand_r)].reshape(1,x_dim*x_dim)
            x_train_trans = np.concatenate([x_train_trans, x_trans])
        train_rot = np.concatenate([train_rot, [i+1]*n_train])
        y_train_trans = np.concatenate([y_train_trans, y_train])
    train_rot_one_hot = one_hot_convert(train_rot)
    return x_train_trans, y_train_trans, train_rot_one_hot

# def rotate_transformation(x_train, y_train, x_dim, n_trans):
#     train_rot = np.array([i for i in range(len(x_train)) for _ in range(n_trans)])
#     x_train_trans = []
#     y_train_trans = []
#     for x, y in zip(x_train, y_train):
#         x_reshape = x.reshape(x_dim,x_dim)
#         x_trans_1 = np.rot90(x_reshape, 1).reshape(1,1024)
#         x_trans_2 = np.rot90(x_reshape, 2).reshape(1,1024)
#         x_trans_3 = np.rot90(x_reshape, 3).reshape(1,1024)
#         x = x.reshape(1,1024)
#         if x_train_trans == []:
#             x_train_trans = np.concatenate([x, x_trans_1, x_trans_2, x_trans_3])
#         else:
#             x_train_trans = np.concatenate([x_train_trans, x, x_trans_1, x_trans_2, x_trans_3])
#         if y_train_trans == []:
#             y_train_trans = np.concatenate([[y], [y], [y], [y]])
#         else:
#             y_train_trans = np.concatenate([y_train_trans, [y], [y], [y], [y]])
#     train_rot_one_hot = one_hot_convert(train_rot)
#     return x_train_trans, y_train_trans, train_rot_one_hot

def ae_random_order_transformation(x_train, y_train, x_dim, n_trans):
    n_train, n_dims = x_train.shape
  #  ws = np.random.randn(n_rots, n_dims, n_dims)
   # bs_train = np.random.randn(n_rots, n_train, n_dims)
    train_rot = x_train
    x_train_trans = x_train
    y_train_trans = y_train
    r = list(range(x_dim))
    random.seed(datetime.now())
    for i in range(n_trans):
        rand_r = random.sample(r, len(r))
        print(rand_r)
        for x in x_train:
            x_trans = x.reshape(x_dim,x_dim)[np.ix_(rand_r, rand_r)].reshape(1,x_dim*x_dim)
            x_train_trans = np.concatenate([x_train_trans, x_trans])
        train_rot = np.concatenate([train_rot, x_train])
        y_train_trans = np.concatenate([y_train_trans, y_train])
    return x_train_trans, y_train_trans, train_rot

def ae_shift_transformation(x_train, y_train, x_dim, n_trans):
    n_train, n_dims = x_train.shape
  #  ws = np.random.randn(n_rots, n_dims, n_dims)
   # bs_train = np.random.randn(n_rots, n_train, n_dims)
    train_rot = x_train
    x_train_trans = x_train
    y_train_trans = y_train
    for i in range(n_trans):
        num = random.randint(0, x_dim-1)
        print(num)
        for k, x in enumerate(x_train):
            x_t = np.roll(x.reshape(x_dim,x_dim), num, 0)
            x_trans = np.roll(x_t, num, 1).reshape(1,x_dim*x_dim)
            x_train_trans = np.concatenate([x_train_trans, x_trans])
        train_rot = np.concatenate([train_rot, x_train])
        y_train_trans = np.concatenate([y_train_trans, y_train])
    return x_train_trans, y_train_trans, train_rot

def ae_swap_transformation(x_train, y_train, x_dim, n_trans):
    n_train, n_dims = x_train.shape
  #  ws = np.random.randn(n_rots, n_dims, n_dims)
   # bs_train = np.random.randn(n_rots, n_train, n_dims)
    train_rot = x_train
    x_train_trans = x_train
    y_train_trans = y_train
    for i in range(n_trans):
        n1, n2 = random.sample(range(0, x_dim), 2)
        print(n1, n2)
        for k, x in enumerate(x_train):
            x_reshape = x.reshape(x_dim,x_dim)
            x_reshape[[n1, n2], :] = x_reshape[[n2, n1], :]
            x_reshape[:, [n1, n2]] = x_reshape[:, [n2, n1]]
            x_train_trans = np.concatenate([x_train_trans, x_reshape.reshape(1,x_dim*x_dim)])
        train_rot = np.concatenate([train_rot, x_train])
        y_train_trans = np.concatenate([y_train_trans, y_train])
    return x_train_trans, y_train_trans, train_rot

def ae_gaussian_transformation(x_train, y_train, x_dim, n_trans):
    n_train, n_dims = x_train.shape
    train_rot = x_train
    mean = 0.0
    std = 0.01
    x_train_trans = x_train
    y_train_trans = y_train
    for i in range(1, n_trans + 1):
        g = np.random.normal(mean, std, (1, x_dim))
        print("start transforming: " + str(i))
        for k, x in enumerate(x_train):
            x_trans = x.reshape(1, x_dim) + g
            x_train_trans = np.concatenate([x_train_trans, x_trans])
        train_rot = np.concatenate([train_rot, x_train])
        y_train_trans = np.concatenate([y_train_trans, y_train])
 #   train_rot_one_hot = one_hot_convert(train_rot)
    return x_train_trans, y_train_trans, train_rot

def ae_affine_transformation(x_train, y_train, x_dim, n_trans):
    n_train, n_dims = x_train.shape
    train_rot = x_train
    x_train_trans = x_train
    y_train_trans = y_train
    for i in range(1, n_trans + 1):
        w = np.random.rand(x_dim, x_dim)
        b = np.random.rand(1, x_dim)
        print("start transforming: " + str(i))
        for k, x in enumerate(x_train):
            x_trans = np.dot(x.reshape(1, x_dim), w) + b
            x_train_trans = np.concatenate([x_train_trans, x_trans])
        train_rot = np.concatenate([train_rot, x_train])
        y_train_trans = np.concatenate([y_train_trans, y_train])
   # train_rot_one_hot = one_hot_convert(train_rot)
    return x_train_trans, y_train_trans, train_rot


def ae_rotate_transformation(x_train, y_train, x_dim, n_trans):
    n_train, n_dims = x_train.shape
    train_rot = x_train
    x_train_trans = x_train
    y_train_trans = y_train
    for i in range(1, n_trans + 1):
        print("start rotating: " + str(i))
        for k, x in enumerate(x_train):
            x_trans = np.rot90(x.reshape(x_dim, x_dim), i).reshape(1, x_dim*x_dim)
            x_train_trans = np.concatenate([x_train_trans, x_trans])
        train_rot = np.concatenate([train_rot, x_train])
        y_train_trans = np.concatenate([y_train_trans, y_train])
   # train_rot_one_hot = one_hot_convert(train_rot)
    return x_train_trans, y_train_trans, train_rot

def ae_img_misc_transformation(x_train, y_train, x_dim):
    n_train, n_dims = x_train.shape
  #  ws = np.random.randn(n_rots, n_dims, n_dims)
   # bs_train = np.random.randn(n_rots, n_train, n_dims)
    train_rot = []
    x_train_trans = []
    y_train_trans = []
    mean = 0
    var = 0.001
    sigma = var ** 0.5
    def patch(img, dim):
        patch_size = 5
        x_rnd = np.random.randint(low=patch_size, high=dim - patch_size)
        y_rnd = np.random.randint(low=patch_size, high=dim - patch_size)
        img1 = cv2.rectangle(img.copy(), (x_rnd, y_rnd), (x_rnd + patch_size, y_rnd + patch_size), (0, 0, 0), -1)
        return img1
    for x, y, i in zip(x_train, y_train, range(len(x_train))):
        if i % 1000 == 0:
            print("No. ", i)
        g = np.random.normal(mean, sigma, (1, x_dim*x_dim))
        x_trans_1 = np.flip(x.reshape(x_dim,x_dim), 1).reshape(1,x_dim*x_dim)
        x_trans_2 = np.rot90(x.reshape(x_dim, x_dim), 1).reshape(1, x_dim*x_dim)
        x_trans_3 = x.reshape(1,x_dim*x_dim)
        x_trans_4 = patch(x.reshape(x_dim, x_dim), x_dim).reshape(1,x_dim*x_dim)
        if len(x_train_trans) == 0:
            x_train_trans = np.concatenate([x_trans_1, x_trans_2, x_trans_3, x_trans_4])
        else:
            x_train_trans = np.concatenate([x_train_trans, x_trans_1, x_trans_2, x_trans_3, x_trans_4])
        if len(train_rot) == 0:
            train_rot = [x, x, x, x]
        else:
            train_rot = np.concatenate([train_rot, [x, x, x, x]])
        if len(y_train_trans) == 0:
            y_train_trans = [y, y, y, y]
        else:
            y_train_trans = np.concatenate([y_train_trans, [y, y, y, y]])
    return x_train_trans, y_train_trans, train_rot

def ae_shift_rotate_transformation(x_train, y_train, x_dim, n_trans):
    n_train, n_dims = x_train.shape
    train_rot = []
    x_train_trans = []
    y_train_trans = []
    for i in range(n_trans):
        print("start rotating: " + str(i))
        for k, x in enumerate(x_train):
            x_trans = np.rot90(x.reshape(x_dim, x_dim), i).reshape(1, x_dim*x_dim)
            if len(x_train_trans) == 0:
                x_train_trans = x_trans
            else:
                x_train_trans = np.concatenate([x_train_trans, x_trans])
            x_trans_rot = np.rot90(x.reshape(x_dim, x_dim), i+1).reshape(1, x_dim*x_dim)
            if len(train_rot) == 0:
                train_rot = x_trans_rot
            else:
                train_rot = np.concatenate([train_rot, x_trans_rot])
        if len(y_train_trans) == 0:
            y_train_trans = y_train
        else:
            y_train_trans = np.concatenate([y_train_trans, y_train])
   # train_rot_one_hot = one_hot_convert(train_rot)
    return x_train_trans, y_train_trans, train_rot

# def rotate_transformation(x_train, y_train, x_dim, n_trans):
#     n_train, n_dims = x_train.shape
#     train_rot = range(n_train)
#     x_train_trans = x_train
#     y_train_trans = y_train
#     for i in range(1, n_trans + 1):
#         print("start rotating: " + str(i))
#         for k, x in enumerate(x_train):
#             x_trans = np.rot90(x.reshape(x_dim, x_dim), i).reshape(1, x_dim*x_dim)
#             x_train_trans = np.concatenate([x_train_trans, x_trans])
#         train_rot = np.concatenate([train_rot, range(n_train)])
#         y_train_trans = np.concatenate([y_train_trans, y_train])
#     train_rot_one_hot = one_hot_convert(train_rot)
#     return x_train_trans, y_train_trans, train_rot_one_hot

def shift_transformation(x_train, y_train, x_dim, n_trans):
    n_train, n_dims = x_train.shape
  #  ws = np.random.randn(n_rots, n_dims, n_dims)
   # bs_train = np.random.randn(n_rots, n_train, n_dims)
    train_rot = [0] * n_train
    x_train_trans = x_train
    y_train_trans = y_train
    for i in range(1, n_trans + 1):
        num = random.randint(0, x_dim-1)
        print(num)
        for k, x in enumerate(x_train):
            x_t = np.roll(x.reshape(x_dim,x_dim), num, 0)
            x_trans = np.roll(x_t, num, 1).reshape(1,x_dim*x_dim)
            x_train_trans = np.concatenate([x_train_trans, x_trans])
        train_rot = np.concatenate([train_rot, [i] * n_train])
        y_train_trans = np.concatenate([y_train_trans, y_train])
    train_rot_one_hot = one_hot_convert(train_rot)
    return x_train_trans, y_train_trans, train_rot_one_hot


def rotate_transformation(x_train, y_train, x_dim, n_trans):
    n_train, n_dims = x_train.shape
    train_rot = [0] * n_train
    x_train_trans = x_train
    y_train_trans = y_train
    for i in range(1, n_trans + 1):
        print("start rotating: " + str(i))
        for k, x in enumerate(x_train):
            x_trans = np.rot90(x.reshape(x_dim, x_dim), i).reshape(1, x_dim*x_dim)
            x_train_trans = np.concatenate([x_train_trans, x_trans])
        train_rot = np.concatenate([train_rot, [i] * n_train])
        y_train_trans = np.concatenate([y_train_trans, y_train])
    train_rot_one_hot = one_hot_convert(train_rot)
    return x_train_trans, y_train_trans, train_rot_one_hot

def crop_transformation(x_train, y_train, x_dim, n_trans):
    n_train, n_dims = x_train.shape
    train_rot = [0] * n_train
    x_train_trans = x_train
    y_train_trans = y_train
    crop_range = 5
    for i in range(1, n_trans + 1):
        rand_int = random.randint(0, x_dim-1)
        print("start transforming: " + str(i))
        for k, x in enumerate(x_train):
            x_trans = np.array([x[j] if j not in range(rand_int, rand_int+crop_range) else 0 for j in range(x_dim)]).reshape(1,x_dim)
            x_train_trans = np.concatenate([x_train_trans, x_trans])
        train_rot = np.concatenate([train_rot, [i] * n_train])
        y_train_trans = np.concatenate([y_train_trans, y_train])
    train_rot_one_hot = one_hot_convert(train_rot)
    return x_train_trans, y_train_trans, train_rot_one_hot

def gaussian_transformation(x_train, y_train, x_dim, n_trans):
    n_train, n_dims = x_train.shape
    train_rot = [0] * n_train
    mean = 0.0
    std = 0.01
    x_train_trans = x_train
    y_train_trans = y_train
    for i in range(1, n_trans + 1):
        g = np.random.normal(mean, std, (1, x_dim*x_dim))
        print("start transforming: " + str(i))
        for k, x in enumerate(x_train):
            x_trans = x.reshape(1, x_dim*x_dim) + g
            x_train_trans = np.concatenate([x_train_trans, x_trans])
        train_rot = np.concatenate([train_rot, [i] * n_train])
        y_train_trans = np.concatenate([y_train_trans, y_train])
    train_rot_one_hot = one_hot_convert(train_rot)
    return x_train_trans, y_train_trans, train_rot_one_hot

def offset_transformation(x_train, y_train, x_dim, n_trans):
    n_train, n_dims = x_train.shape
  #  ws = np.random.randn(n_rots, n_dims, n_dims)
   # bs_train = np.random.randn(n_rots, n_train, n_dims)
    train_rot = [0] * n_train
    x_train_trans = x_train
    y_train_trans = y_train
    for i in range(n_trans):
        print("start transforming: " + str(i))
        num = random.randint(0, x_dim)
        for k, x in enumerate(x_train):
            x_trans = np.roll(x.reshape(x_dim, x_dim), num, axis=(0, 1)).reshape(1, x_dim*x_dim)
            x_train_trans = np.concatenate([x_train_trans, x_trans])
        train_rot = np.concatenate([train_rot, [i+1]*n_train])
        y_train_trans = np.concatenate([y_train_trans, y_train])
    train_rot_one_hot = one_hot_convert(train_rot)
    return x_train_trans, y_train_trans, train_rot_one_hot

def affine_transformation(x_train, y_train, x_dim, n_trans):
    n_train, n_dims = x_train.shape
    train_rot = [0] * n_train
    x_train_trans = x_train
    y_train_trans = y_train
    for i in range(1, n_trans + 1):
        w = np.random.rand(x_dim*x_dim, x_dim*x_dim)
        b = np.random.rand(1, x_dim*x_dim)
        print("start transforming: " + str(i))
        for k, x in enumerate(x_train):
            x_trans = np.dot(x.reshape(1, x_dim*x_dim), w) + b
            x_train_trans = np.concatenate([x_train_trans, x_trans])
        train_rot = np.concatenate([train_rot, [i] * n_train])
        y_train_trans = np.concatenate([y_train_trans, y_train])
    train_rot_one_hot = one_hot_convert(train_rot)
    return x_train_trans, y_train_trans, train_rot_one_hot

def load_open_dataset(dataset_name, tr_classes, seed, rot=None, normalize=False):
    if dataset_name == 'ms':
        # Load Data
        with open('./data/pickle/5-bit-cluster_32-bit_minhash_1_1-gram_all-files_run1.pkl') as fin:
            data_fcg_xs, data_fcg_ys = pickle.load(fin)
        return OpenWorldMsData(data_fcg_xs, data_fcg_ys, tr_classes=tr_classes,
                               comb_val_test=False, seed=seed, normalize=normalize)

    elif dataset_name == 'msadjmat':
        with open('./data/fcg_adj_mat/5-bit-cluster_32-bit_minhash_1_1-gram_all-files_run1_adj_matrix.pkl') as fin:
            fcg_adj_xs, fcg_adj_ys = pickle.load(fin)

        # Flatten
        fcg_adj_xs = np.reshape(fcg_adj_xs, [fcg_adj_xs.shape[0], reduce(lambda x, y: x*y, fcg_adj_xs.shape[1:])])
        # One hot encode
        enc = preprocessing.OneHotEncoder(n_values=9, sparse=False)
        enc.fit(fcg_adj_ys.reshape(-1, 1))
        fcg_adj_ys = enc.transform(fcg_adj_ys.reshape(-1, 1))

        return OpenWorldMsData(fcg_adj_xs, np.argmax(fcg_adj_ys, axis=1),
                               tr_classes=6, comb_val_test=False, seed=1, normalize=normalize)

    elif dataset_name == 'android':
        with open('./data/pickle/android_malware_genome_project_5-bit-cluster_32-bit_minhash-1_1-gram.pkl') as fin:
            android_fcg_xs, android_fcg_ys = pickle.load(fin)
        uvalues, ucounts =  np.unique(android_fcg_ys, return_counts=True)
        min_class_size = 40  ## Smallest class size allowed
        true_classes = set(uvalues[ucounts >= min_class_size])
        mask = [False] * len(android_fcg_ys)
        for i in xrange(len(android_fcg_ys)):
            if android_fcg_ys[i] in true_classes:
                mask[i] = True
        mask = np.array(mask)

        # Only consider classes which have > min_class_size of number instance per class
        android_fcg_xs = android_fcg_xs[mask]
        android_fcg_ys = android_fcg_ys[mask]

        # Remap class labels to range between 0-8
        android_label_lookup = {c:i for i, c in enumerate(np.unique(android_fcg_ys))}
        for i in xrange(len(android_fcg_ys)):
            android_fcg_ys[i] = android_label_lookup[android_fcg_ys[i]]

        return OpenWorldMsData(android_fcg_xs, android_fcg_ys,
                               tr_classes=tr_classes,
                               comb_val_test=False, seed=seed, normalize=normalize)
    elif dataset_name == 'mnist':
        mnist = input_data.read_data_sets("data/MNIST_data/", one_hot=True)
        return OpenWorldSim(mnist.train.images, mnist.train.labels,
                            val_data=mnist.validation.images, val_label=mnist.validation.labels,
                            test_data=mnist.test.images, test_label=mnist.test.labels,
                            tr_classes=tr_classes, seed=seed)

    elif dataset_name == 'fashion-mnist':
        num_classes = 10
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        x_train = []
        x_test = []

        for i in range(len(train_images)):
            x_train.append(train_images[i].flatten())

        for i in range(len(test_images)):
            x_test.append(test_images[i].flatten())


        x_train = np.array(x_train)
        x_test = np.array(x_test)

        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255


        y_train = keras.utils.to_categorical(train_labels, num_classes)
        y_test = keras.utils.to_categorical(test_labels, num_classes)
        return OpenWorldSim(x_train, y_train,
                            val_data=x_test, val_label=y_test,
                            test_data=x_test, test_label=y_test,
                            tr_classes=tr_classes, seed=seed)

    elif dataset_name == 'stl10':
        # path to the binary train file with image data
        DATA_PATH = './data/stl10_binary/train_X.bin'
        # path to the binary train file with labels
        LABEL_PATH = './data/stl10_binary/train_y.bin'
        TEST_DATA = './data/stl10_binary/test_X.bin'
        TEST_LABEL = './data/stl10_binary/test_y.bin'
        stl10_input.download_and_extract()

        num_classes = 10
        x_train_rgb = stl10_input.read_all_images(DATA_PATH)
        y_train_org = stl10_input.read_labels(LABEL_PATH)
        x_test_rgb = stl10_input.read_all_images(TEST_DATA)
        y_test_org = stl10_input.read_labels(TEST_LABEL)

        x_train = []
        x_test = []

        for i in range(len(x_train_rgb)):
            x_train.append(cv2.cvtColor(x_train_rgb[i], cv2.COLOR_BGR2GRAY).flatten())

        for i in range(len(x_test_rgb)):
            x_test.append(cv2.cvtColor(x_test_rgb[i], cv2.COLOR_BGR2GRAY).flatten())

        x_train = np.array(x_train)
        x_test = np.array(x_test)

        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        y_train = [value - 1 for value in y_train_org]
        y_test = [value - 1 for value in y_test_org]
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
     #   print(x_train.shape, y_train.shape)
        return OpenWorldSim(x_train, y_train,
                            val_data=x_test, val_label=y_test,
                            test_data=x_test, test_label=y_test,
                            tr_classes=tr_classes, seed=seed)

    elif dataset_name == 'cifar100':
        num_classes = 100
        (x_train_rgb, y_train), (x_test_rgb, y_test) = cifar100.load_data()
        x_train = []
        x_test = []

        for i in range(len(x_train_rgb)):
            x_train.append(cv2.cvtColor(x_train_rgb[i], cv2.COLOR_BGR2GRAY).flatten())

        for i in range(len(x_test_rgb)):
            x_test.append(cv2.cvtColor(x_test_rgb[i], cv2.COLOR_BGR2GRAY).flatten())

        x_train = np.array(x_train)
        x_test = np.array(x_test)

        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        return OpenWorldSim(x_train, y_train,
                            val_data=x_test, val_label=y_test,
                            test_data=x_test, test_label=y_test,
                            tr_classes=tr_classes, seed=seed)


    elif dataset_name == 'cifar10':
        num_classes = 10
        (x_train_rgb, y_train), (x_test_rgb, y_test) = cifar10.load_data()
        x_train = []
        x_test = []

        for i in range(len(x_train_rgb)):
            x_train.append(cv2.cvtColor(x_train_rgb[i], cv2.COLOR_BGR2GRAY).flatten())

        for i in range(len(x_test_rgb)):
            x_test.append(cv2.cvtColor(x_test_rgb[i], cv2.COLOR_BGR2GRAY).flatten())

        x_train = np.array(x_train)
        x_test = np.array(x_test)

        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        return OpenWorldSim(x_train, y_train,
                            val_data=x_test, val_label=y_test,
                            test_data=x_test, test_label=y_test,
                            tr_classes=tr_classes, seed=seed)

    elif dataset_name == 'svhn':
        svhn_train = scipy.io.loadmat('data/SVHN_data/train_32x32.mat')
        svhn_test = scipy.io.loadmat('data/SVHN_data/test_32x32.mat')
        svhn_extra = scipy.io.loadmat('data/SVHN_data/extra_32x32.mat')

        def rotate_axis(X):
            X = np.swapaxes(X,2,3)
            X = np.swapaxes(X,1,2)
            X = np.swapaxes(X,0,1)
            return X

        def flatten(X):
            return X.reshape((-1, reduce((lambda x, y: x*y), X.shape[1:])))

        svhn_train['X'] = flatten(rotate_axis(svhn_train['X']))
        svhn_extra['X'] = flatten(rotate_axis(svhn_extra['X']))
        svhn_test['X'] = flatten(rotate_axis(svhn_test['X']))

        enc = preprocessing.OneHotEncoder( sparse=False) # n_values=10,
        enc.fit(svhn_train['y'].reshape(-1, 1))

        svhn_train['y'] = enc.transform(svhn_train['y'].reshape(-1, 1))
        svhn_extra['y'] = enc.transform(svhn_extra['y'].reshape(-1, 1))
        svhn_test['y'] = enc.transform(svhn_test['y'].reshape(-1, 1))

        return OpenWorldSim(svhn_train['X'], svhn_train['y'],
                            val_data=svhn_extra['X'][-10000:],
                            val_label=svhn_extra['y'][-10000:],
                            test_data=svhn_test['X'], test_label=svhn_test['y'],
                            tr_classes=tr_classes, seed=seed)
