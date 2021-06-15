import sys
import os.path
sys.path.insert(0, os.path.abspath("./simple-dnn"))

#Import the libraries we will need.
from IPython.display import display
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow.contrib.slim as slim
import scipy.misc
import scipy
import scipy.io
from sklearn import metrics, preprocessing
from sklearn.neighbors import KernelDensity
import time
import pickle
import cPickle
import matplotlib.cm as cm
import random
import statistics

from util.openworld_sim import OpenWorldSim, OpenWorldMsData
from util.visualization import visualize_dataset_2d
from simple_dnn.cnn.dcnn import DCNN
from simple_dnn.util.format import UnitPosNegScale, reshape_pad
from simple_dnn.generative.vae import VariationalAutoencoder
from simple_dnn.generative.gan import MultiClassGAN
from simple_dnn.generative.discriminator import DiscriminatorDC
from simple_dnn.generative.generator import GeneratorDC
from simple_dnn.util.sample_writer import ImageGridWriter

from open_net import OpenNetFlat, OpenNetCNN, OpenNetBase
from util.metrics import auc, open_set_classification_metric, open_classification_performance
from util.open_net_train_eval import train_eval, compare_performance, ttest
from util.visualization import visualize_dataset_nd

import argparse
import subprocess
import random
import sys

mnist = input_data.read_data_sets("data/MNIST_data/", one_hot=True)
#zd = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
comb = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#comb = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
tr_classes_list = []
with open("/home/jiaj2018/code/opennet_ii/data/mini_mnist") as fin:
    for line in fin:
        if line.strip() == '':
            continue
        cols = line.strip().split()
        tr_classes_list.append([int(float(c)) for c in cols])

cor_acc_comb = []
for c in comb:
    cor_acc_mul = []
    for tr_classes in tr_classes_list:
        length = 6
        acc = 0
        label_text_lookup = {i: str(c) for i, c in enumerate(sorted(tr_classes))}
        label_text_lookup[6] = 'unknown'

        open_mnist = OpenWorldSim(mnist.train.images, mnist.train.labels,
                         val_data=mnist.validation.images, val_label=mnist.validation.labels,
                         test_data=mnist.test.images, test_label=mnist.test.labels,
                         tr_classes=[0,2,3,4,6,9],
                         seed=None)

        print("comb coefficient: " + str(c))
        with tf.device('/GPU:0'):
            cnn_disc_ae_6_long = OpenNetCNN(
            [32, 32],  # x_dim
            1,  #x_ch
            length,  #y_dim
            [32, 64], # conv_units,
            [256, 128],      #hidden_units
            z_dim=6,
            kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
            pooling_enable=True, pooling_kernel=[3,3],
            pooling_stride=[2,2], pooling_padding='SAME',
            pooling_type='max',
            activation_fn=tf.nn.relu,

            x_scale=UnitPosNegScale.scale,
            x_inverse_scale=UnitPosNegScale.inverse_scale,
            x_reshape=reshape_pad([28,28], [32,32], 1, pad=True, pad_value=-1),

            opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
            recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
            c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),

            dist='mean_separation_spread',
            decision_dist_fn='mahalanobis',#'euclidean',
            dropout=True, keep_prob=0.2,
#           batch_norm=True,
            batch_size=256,
            iterations=5000,
            display_step=1000,
            save_step=500,
            model_directory=None,  # Directory to save trained model to.
            density_estimation_factory=None,
            ce_loss=False, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False, tc_loss=False, cor_loss=True,
            contamination=0.01, comb=c)

        acc, _, _ = train_eval(cnn_disc_ae_6_long, open_mnist.train_data(), open_mnist.train_label(),
            open_mnist.validation_data(), open_mnist.validation_label(),
            np.logical_not(open_mnist.validation_label()[:,-1].astype(bool)),
            open_mnist.test_data(), open_mnist.test_label(),
            np.logical_not(open_mnist.test_label()[:,-1].astype(bool)),
            n_scatter=1000, unique_ys=range(7), plot_recon=False,
#                save_path='data/results/fig/mnist_cnn_z6_plot.pdf',
            label_text_lookup=label_text_lookup, visualize=False, acc=acc
            )

        cor_acc_mul.append(acc)
    cor_acc_comb.append(sum(cor_acc_mul) / len(tr_classes_list) )
    #cor_acc_comb.append(statistics.mean(cor_acc_mul))

print("pred acc under different coefficients:")
print(cor_acc_comb)



