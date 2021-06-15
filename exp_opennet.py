import argparse
import logging

import sys
import os.path
import numpy as np
import tensorflow as tf
import time
import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

sys.path.insert(0, os.path.abspath("./simple-dnn"))

from g_openmax import GOpenmax
from open_net import OpenNetFlat, OpenNetCNN
from openmax import OpenMaxFlat, OpenMaxCNN
from central_opennet import CentralOpennetFlat, CentralOpennetCNN
from exp_opennet_util import load_open_dataset, save_pickle_gz, random_order_transformation, \
    rotate_transformation, affine_transformation, crop_transformation, gaussian_transformation, offset_transformation, \
    ae_rotate_transformation, ae_shift_rotate_transformation, ae_img_misc_transformation, ae_random_order_transformation, \
    ae_gaussian_transformation, ae_affine_transformation, ae_shift_transformation, ae_swap_transformation, shift_transformation
from simple_dnn.generative.discriminator import DiscriminatorDC, DiscriminatorFlat
from simple_dnn.generative.gan import MultiClassGAN, FlatGAN
from simple_dnn.generative.generator import GeneratorDC, GeneratorFlat
from simple_dnn.util.sample_writer import ImageGridWriter
from simple_dnn.util.format import UnitPosNegScale, reshape_pad, flatten_unpad
from util.openworld_sim import OpenWorldSim, OpenWorldMsData

# Open Models
def get_flat_model_factories(model_name, dataset_name, open_dataset, z_dim=None, model_directory=None, load_model_directory=None):
    if dataset_name == 'mnist':
        if model_name == 'mmf':
            return lambda : OpenNetFlat(
                open_dataset.train_data().shape[1], # x_dim
                y_dim=6,
                z_dim=6 if z_dim is None else z_dim,
                h_dims=[512, 128],
                dropout = True, keep_prob=0.25,
                dist='mean_separation_spread',#'min_max',#'mean_separation_spread',#'class_mean',#
                decision_dist_fn = 'mahalanobis',
                batch_size=128,
                iterations=20000,
                display_step=-1,
                save_step=1000,
                ce_loss=False, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False, mmf_extension=True,
                contamination=0.01,
                activation_fn=tf.nn.relu, # LeakyReLUfunction, #
                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
            )
        elif model_name == 'ii':
            return lambda : OpenNetFlat(
                open_dataset.train_data().shape[1], # x_dim
                y_dim=6,
                z_dim=6 if z_dim is None else z_dim,
                h_dims=[512, 128],
                dropout = True, keep_prob=0.25,
                dist='mean_separation_spread',#'min_max',#'mean_separation_spread',#'class_mean',#
                decision_dist_fn = 'mahalanobis',
                batch_size=128,
                iterations=20000,
                display_step=-1,
                save_step=1000,
                ce_loss=False, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False,
                contamination=0.01,
                activation_fn=tf.nn.relu, # LeakyReLUfunction, #
                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
            )
        elif model_name == 'ce':
            return lambda : OpenNetFlat(
                open_dataset.train_data().shape[1], # x_dim
                y_dim=6,
                z_dim=6 if z_dim is None else z_dim,
                h_dims=[512, 128],
                dropout = True, keep_prob=0.25,
                dist='mean_separation_spread',#'class_mean',#
                decision_dist_fn = 'mahalanobis',
                batch_size=128,
                iterations=10000,
                display_step=-1,
                ce_loss=True, recon_loss=False, inter_loss=False, intra_loss=False, div_loss=False,
                contamination=0.01,
                activation_fn=tf.nn.relu, # LeakyReLUfunction, #
                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
            )
        elif model_name == 'ceii':
            return lambda : OpenNetFlat(
                open_dataset.train_data().shape[1], # x_dim
                y_dim=6,
                z_dim=6 if z_dim is None else z_dim,
                h_dims=[512, 128],
                dropout = True, keep_prob=0.25,
                dist='mean_separation_spread',#'class_mean',#
                decision_dist_fn = 'mahalanobis',
                batch_size=128,
                iterations=10000,
                display_step=-1,
                ce_loss=True, recon_loss=False, inter_loss=False, intra_loss=True, div_loss=False,
                contamination=0.01,
                activation_fn=tf.nn.relu, # LeakyReLUfunction, #
                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
            )
        elif model_name == 'openmax':
            return lambda : OpenMaxFlat(
                open_dataset.train_data().shape[1], # x_dim
                y_dim=6,
                h_dims=[512, 128],
                dropout = True, keep_prob=0.25,
                decision_dist_fn = 'eucos',
                batch_size=128,
                iterations=10000,
                display_step=-1,
                activation_fn=tf.nn.relu, # LeakyReLUfunction, #
                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                tailsize = 20,
                alpharank = 4,
            )

def get_cnn_model_factories(model_name, dataset_name, open_dataset, z_dim=None, model_directory=None, load_model_directory=None):

    if dataset_name == 'mnist':
        if model_name == 'recon-self-supervision':
            return lambda : OpenNetCNN(
                [32, 32],  # x_dim
                1,  #x_chs
                6,  #y_dim
                [32, 64], # conv_units,
                [256, 128],      #hidden_units
                z_dim=6 if z_dim is None else z_dim,
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
                rot_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),

                dist='triplet_center_loss',
                decision_dist_fn='mahalanobis',
                dropout = True, keep_prob=0.2,
                batch_size=256,
                iterations=15000,
                display_step=-500,
                save_step=1000,
                model_directory=model_directory,  # Directory to save trained model to.
                density_estimation_factory=None,
                ce_loss=False, recon_loss=True, inter_loss=False, intra_loss=False, div_loss=False, mmf_extension=False, tc_loss=False,
                contamination=0.01, margin=2.0, n_rots=[32, 32], self_sup = False
            )
        elif model_name == 'self-supervision':
            return lambda : OpenNetCNN(
                [32, 32],  # x_dim
                1,  #x_chs
                6,  #y_dim
                [32, 64], # conv_units,
                [256, 128],      #hidden_units
                z_dim=6 if z_dim is None else z_dim,
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
                rot_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),

                dist='triplet_center_loss',
                decision_dist_fn='mahalanobis',
                dropout = True, keep_prob=0.2,
                batch_size=256,
                iterations=15000,
                display_step=-500,
                save_step=1000,
                model_directory=model_directory,  # Directory to save trained model to.
                density_estimation_factory=None,
                ce_loss=False, recon_loss=False, inter_loss=False, intra_loss=False, div_loss=False, mmf_extension=False, tc_loss=False,
                contamination=0.01, margin=2.0, n_rots=4, self_sup = True
            )
        if model_name == 'triplet':
            return lambda : OpenNetCNN(
                [32, 32],  # x_dim
                1,  #x_chs
                6,  #y_dim
                [32, 64], # conv_units,
                [256, 128],      #hidden_units
                z_dim=6 if z_dim is None else z_dim,
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

                dist='triplet_center_loss',
                decision_dist_fn='mahalanobis',
                dropout = True, keep_prob=0.2,
                batch_size=256,
                iterations=5000,
                display_step=-500,
                save_step=500,
                model_directory=model_directory,  # Directory to save trained model to.
                load_model_directory=load_model_directory,
                density_estimation_factory=None,
                ce_loss=False, recon_loss=False, inter_loss=False, intra_loss=False, div_loss=False, mmf_extension=False, tc_loss=True,
                contamination=0.01, margin=2.0
            )
        elif model_name == 'ii':
            return lambda : OpenNetCNN(
                [32, 32],  # x_dim
                1,  #x_ch
                6,  #y_dim
                [32, 64], # conv_units,
                [256, 128],      #hidden_units
                z_dim=6 if z_dim is None else z_dim,
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
                decision_dist_fn='mahalanobis',
                dropout = True, keep_prob=0.2,
                batch_size=256,
                iterations=5000,
                display_step=-500,
                save_step=500,
                model_directory=model_directory,  # Directory to save trained model to.
                load_model_directory=load_model_directory,

                density_estimation_factory=None,
                ce_loss=False, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False,  mmf_extension=False,
                contamination=0.01
            )
        elif model_name == 'ce':
            return lambda : OpenNetCNN(
                [32, 32],  # x_dim
                1,  #x_ch
                6,  #y_dim
                [32, 64], # conv_units,
                [256, 128],      #hidden_units
                z_dim=6 if z_dim is None else z_dim,
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
                decision_dist_fn='mahalanobis',
                dropout = True, keep_prob=0.2,
                batch_size=256,
                iterations=5000,
                display_step=-500,
                save_step=500,
                model_directory=model_directory,  # Directory to save trained model to.
                load_model_directory=load_model_directory,
                density_estimation_factory=None,
                ce_loss=True, recon_loss=False, inter_loss=False, intra_loss=False, div_loss=False,
                contamination=0.01,
            )
        elif model_name == 'ceii':
            return lambda : OpenNetCNN(
                [32, 32],  # x_dim
                1,  #x_ch
                6,  #y_dim
                [32, 64], # conv_units,
                [256, 128],      #hidden_units
                z_dim=6 if z_dim is None else z_dim,
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
                decision_dist_fn='mahalanobis',
                dropout = True, keep_prob=0.2,
                batch_size=256,
                iterations=5000,
                display_step=-500,
                save_step=500,
                model_directory=None,  # Directory to save trained model to.
                density_estimation_factory=None,
                ce_loss=True, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False,
                contamination=0.01,
            )
        elif model_name == 'central':
            return lambda : CentralOpennetCNN(
                [32, 32],  # x_dim
                1,  #x_ch
                6,  #y_dim
                [32, 64], # conv_units,
                [256, 128],      #hidden_units
                kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                pooling_enable=True, pooling_kernel=[3,3],
                pooling_stride=[2,2], pooling_padding='SAME',
                pooling_type='max',
                activation_fn=tf.nn.relu,

                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                x_reshape=reshape_pad([28,28], [32,32], 1, pad=True, pad_value=-1),

                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                
                decision_dist_fn='mahalanobis',
                dropout = True, keep_prob=0.2,
                batch_size=256,
                iterations=5000,
                display_step=-500,
                save_step=500,
                model_directory=None,  # Directory to save trained model to.
                contamination=0.01,
                penalty=0.01,
            )
        elif model_name == 'openmax':
            return lambda : OpenMaxCNN(
                [32, 32],  # x_dim
                1,  #x_ch
                6,  #y_dim
                [32, 64], # conv_units,
                [256, 128],      #hidden_units
                kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                pooling_enable=True, pooling_kernel=[3,3],
                pooling_stride=[2,2], pooling_padding='SAME',
                pooling_type='max',
                activation_fn=tf.nn.relu,

                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                x_reshape=reshape_pad([28,28], [32,32], 1, pad=True, pad_value=-1),

                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),

                decision_dist_fn='eucos',
                dropout = True, keep_prob=0.2,
                batch_size=256,
                iterations=5000,
                display_step=-500,
                save_step=500,
                model_directory=None,  # Directory to save trained model to.
                tailsize = 20,
                alpharank = 3,
            )
        elif model_name == 'g_openmax':
            return lambda : GOpenmax(
                gan_factory = lambda y_dim: MultiClassGAN(
                        [32, 32], # x_dim 
                        1, # x_ch 
                        y_dim, # y_dim 
                        z_dim=100,
                        generator=GeneratorDC([32, 32],#x_dims
                        1, # x_ch
                        [64,32,16], # g_conv_units
                        g_kernel_sizes=[5,5], g_strides=[2, 2], g_paddings='SAME',
                        g_activation_fn=tf.nn.relu),     # Generator Net
                        discriminator=DiscriminatorDC(6,  # y_dim
                                [16,32,64], # conv_units
                                hidden_units=None,
                                kernel_sizes=[5,5], strides=[2, 2], paddings='SAME',
                                d_activation_fn=tf.contrib.keras.layers.LeakyReLU,
                                f_activation_fns=tf.nn.relu,
                                dropout=False, keep_prob=0.5), # Discriminator Net
                        x_reshape=reshape_pad([28,28], [32,32], 1, pad=True, pad_value=-1),
                        x_scale=UnitPosNegScale.scale,
                        x_inverse_scale=UnitPosNegScale.inverse_scale,
                        d_optimizer=tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5),
                        g_optimizer=tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5),
                        g_loss_fn='default',
                        d_label_smooth=0.75,
                        ## Training config
                        batch_size=128,
                        iterations=5000,
                        display_step=-500,
                        save_step=500,
                        ), 
                openmax_factory= lambda y_dim: OpenMaxCNN(
                        [32, 32],  1,  #x_ch
                        y_dim,  #y_dim
                        [32, 64], # conv_units,
                        [256, 128],      #hidden_units
                        kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                        pooling_enable=True, pooling_kernel=[3,3],
                        pooling_stride=[2,2], pooling_padding='SAME',
                        pooling_type='max',
                        activation_fn=tf.nn.relu,
                        x_scale=UnitPosNegScale.scale,
                        x_inverse_scale=UnitPosNegScale.inverse_scale,
                        x_reshape=reshape_pad([28,28], [32,32], 1, pad=True, pad_value=-1),
                        c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                        decision_dist_fn='eucos',
                        dropout = True, keep_prob=0.2,
                        batch_size=256,
                        iterations=1500,
                        display_step=-500,
                        save_step=500,
                        tailsize = 20,
                        alpharank = 3), 
                classifier_factory = lambda y_dim: OpenMaxCNN(
                        [32, 32],  1,  #x_ch
                        y_dim,  #y_dim
                        [32, 64], # conv_units,
                        [256, 128],      #hidden_units
                        kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                        pooling_enable=True, pooling_kernel=[3,3],
                        pooling_stride=[2,2], pooling_padding='SAME',
                        pooling_type='max',
                        activation_fn=tf.nn.relu,
                        x_scale=UnitPosNegScale.scale,
                        x_inverse_scale=UnitPosNegScale.inverse_scale,
                        x_reshape=reshape_pad([28,28], [32,32], 1, pad=True, pad_value=-1),
                        c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                        decision_dist_fn='eucos',
                        dropout = True, keep_prob=0.2,
                        batch_size=256,
                        iterations=1500,
                        display_step=-500,
                        tailsize = 20,
                        alpharank = 3), 
                y_dim=6, batch_size=128,
                unpad_flatten=flatten_unpad([32,32], [28,28],1))
    elif dataset_name == 'fashion-mnist':
        if model_name == 'recon-self-supervision':
            return lambda : OpenNetCNN(
                [32, 32],  # x_dim
                1,  #x_chs
                6,  #y_dim
                [32, 64], # conv_units,
                [256, 128],      #hidden_units
                z_dim=6 if z_dim is None else z_dim,
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
                rot_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),

                dist='triplet_center_loss',
                decision_dist_fn='mahalanobis',
                dropout = True, keep_prob=0.2,
                batch_size=256,
                iterations=5000,
                display_step=-500,
                save_step=1000,
                model_directory=model_directory,  # Directory to save trained model to.
                density_estimation_factory=None,
                ce_loss=False, recon_loss=True, inter_loss=False, intra_loss=False, div_loss=False, mmf_extension=False, tc_loss=False,
                contamination=0.01, margin=2.0, n_rots=[32, 32], self_sup = False
            )
        elif model_name == 'self-supervision':
            return lambda : OpenNetCNN(
                [32, 32],  # x_dim
                1,  #x_chs
                6,  #y_dim
                [32, 64], # conv_units,
                [256, 128],      #hidden_units
                z_dim=6 if z_dim is None else z_dim,
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
                rot_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),

                dist='triplet_center_loss',
                decision_dist_fn='mahalanobis',
                dropout = True, keep_prob=0.2,
                batch_size=256,
                iterations=5000,
                display_step=-500,
                save_step=1000,
                model_directory=model_directory,  # Directory to save trained model to.
                density_estimation_factory=None,
                ce_loss=False, recon_loss=False, inter_loss=False, intra_loss=False, div_loss=False, mmf_extension=False, tc_loss=False,
                contamination=0.01, margin=2.0, n_rots=4, self_sup = True
            )
        elif model_name == 'triplet':
            return lambda : OpenNetCNN(
                [32, 32],  # x_dim
                1,  #x_chs
                6,  #y_dim
                [32, 64], # conv_units,
                [256, 128],      #hidden_units
                z_dim=6 if z_dim is None else z_dim,
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

                dist='triplet_center_loss',
                decision_dist_fn='mahalanobis',
                dropout = True, keep_prob=0.2,
                batch_size=256,
                iterations=8000,
                display_step=-500,
                save_step=500,
                model_directory=model_directory,  # Directory to save trained model to.
                load_model_directory=load_model_directory,
                density_estimation_factory=None,
                ce_loss=False, recon_loss=False, inter_loss=False, intra_loss=False, div_loss=False, mmf_extension=False, tc_loss=True,
                contamination=0.01, margin=2.0
            )
        elif model_name == 'ii':
            return lambda : OpenNetCNN(
                [32, 32],  # x_dim
                1,  #x_ch
                6,  #y_dim
                [32, 64], # conv_units,
                [256, 128],      #hidden_units
                z_dim=6 if z_dim is None else z_dim,
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
                decision_dist_fn='mahalanobis',
                dropout = True, keep_prob=0.2,
                batch_size=256,
                iterations=8000,
                display_step=-500,
                save_step=500,
                model_directory=model_directory,  # Directory to save trained model to.
                load_model_directory=load_model_directory,
                density_estimation_factory=None,
                ce_loss=False, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False,  mmf_extension=False,
                contamination=0.01
            )
        elif model_name == 'ce':
            return lambda : OpenNetCNN(
                [32, 32],  # x_dim
                1,  #x_ch
                6,  #y_dim
                [32, 64], # conv_units,
                [256, 128],      #hidden_units
                z_dim=6 if z_dim is None else z_dim,
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
                decision_dist_fn='mahalanobis',
                dropout = True, keep_prob=0.2,
                batch_size=256,
                iterations=8000,
                display_step=-500,
                save_step=500,
                model_directory=model_directory,  # Directory to save trained model to.
                load_model_directory=load_model_directory,
                density_estimation_factory=None,
                ce_loss=True, recon_loss=False, inter_loss=False, intra_loss=False, div_loss=False,
                contamination=0.01,
            )
        elif model_name == 'ceii':
            return lambda : OpenNetCNN(
                [32, 32],  # x_dim
                1,  #x_ch
                6,  #y_dim
                [32, 64], # conv_units,
                [256, 128],      #hidden_units
                z_dim=6 if z_dim is None else z_dim,
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
                decision_dist_fn='mahalanobis',
                dropout = True, keep_prob=0.2,
                batch_size=256,
                iterations=5000,
                display_step=-500,
                save_step=500,
                model_directory=None,  # Directory to save trained model to.
                density_estimation_factory=None,
                ce_loss=True, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False,
                contamination=0.01,
            )
        elif model_name == 'central':
            return lambda : CentralOpennetCNN(
                [32, 32],  # x_dim
                1,  #x_ch
                6,  #y_dim
                [32, 64], # conv_units,
                [256, 128],      #hidden_units
                kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                pooling_enable=True, pooling_kernel=[3,3],
                pooling_stride=[2,2], pooling_padding='SAME',
                pooling_type='max',
                activation_fn=tf.nn.relu,

                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                x_reshape=reshape_pad([28,28], [32,32], 1, pad=True, pad_value=-1),

                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),

                decision_dist_fn='mahalanobis',
                dropout = True, keep_prob=0.2,
                batch_size=256,
                iterations=5000,
                display_step=-500,
                save_step=500,
                model_directory=None,  # Directory to save trained model to.
                contamination=0.01,
                penalty=0.01,
            )
        elif model_name == 'openmax':
            return lambda : OpenMaxCNN(
                [32, 32],  # x_dim
                1,  #x_ch
                6,  #y_dim
                [32, 64], # conv_units,
                [256, 128],      #hidden_units
                kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                pooling_enable=True, pooling_kernel=[3,3],
                pooling_stride=[2,2], pooling_padding='SAME',
                pooling_type='max',
                activation_fn=tf.nn.relu,

                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                x_reshape=reshape_pad([28,28], [32,32], 1, pad=True, pad_value=-1),

                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),

                decision_dist_fn='eucos',
                dropout = True, keep_prob=0.2,
                batch_size=256,
                iterations=8000,
                display_step=-500,
                save_step=500,
                model_directory=None,  # Directory to save trained model to.
                tailsize = 20,
                alpharank = 3,
            )
        elif model_name == 'g_openmax':
            return lambda : GOpenmax(
                gan_factory = lambda y_dim: MultiClassGAN(
                        [32, 32], # x_dim
                        1, # x_ch
                        y_dim, # y_dim
                        z_dim=100,
                        generator=GeneratorDC([32, 32],#x_dims
                        1, # x_ch
                        [64,32,16], # g_conv_units
                        g_kernel_sizes=[5,5], g_strides=[2, 2], g_paddings='SAME',
                        g_activation_fn=tf.nn.relu),     # Generator Net
                        discriminator=DiscriminatorDC(6,  # y_dim
                                [16,32,64], # conv_units
                                hidden_units=None,
                                kernel_sizes=[5,5], strides=[2, 2], paddings='SAME',
                                d_activation_fn=tf.contrib.keras.layers.LeakyReLU,
                                f_activation_fns=tf.nn.relu,
                                dropout=False, keep_prob=0.5), # Discriminator Net
                        x_reshape=reshape_pad([28,28], [32,32], 1, pad=True, pad_value=-1),
                        x_scale=UnitPosNegScale.scale,
                        x_inverse_scale=UnitPosNegScale.inverse_scale,
                        d_optimizer=tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5),
                        g_optimizer=tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5),
                        g_loss_fn='default',
                        d_label_smooth=0.75,
                        ## Training config
                        batch_size=128,
                        iterations=5000,
                        display_step=-500,
                        save_step=500,
                        ),
                openmax_factory= lambda y_dim: OpenMaxCNN(
                        [32, 32],  1,  #x_ch
                        y_dim,  #y_dim
                        [32, 64], # conv_units,
                        [256, 128],      #hidden_units
                        kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                        pooling_enable=True, pooling_kernel=[3,3],
                        pooling_stride=[2,2], pooling_padding='SAME',
                        pooling_type='max',
                        activation_fn=tf.nn.relu,
                        x_scale=UnitPosNegScale.scale,
                        x_inverse_scale=UnitPosNegScale.inverse_scale,
                        x_reshape=reshape_pad([28,28], [32,32], 1, pad=True, pad_value=-1),
                        c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                        decision_dist_fn='eucos',
                        dropout = True, keep_prob=0.2,
                        batch_size=256,
                        iterations=1500,
                        display_step=-500,
                        save_step=500,
                        tailsize = 20,
                        alpharank = 3),
                classifier_factory = lambda y_dim: OpenMaxCNN(
                        [32, 32],  1,  #x_ch
                        y_dim,  #y_dim
                        [32, 64], # conv_units,
                        [256, 128],      #hidden_units
                        kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                        pooling_enable=True, pooling_kernel=[3,3],
                        pooling_stride=[2,2], pooling_padding='SAME',
                        pooling_type='max',
                        activation_fn=tf.nn.relu,
                        x_scale=UnitPosNegScale.scale,
                        x_inverse_scale=UnitPosNegScale.inverse_scale,
                        x_reshape=reshape_pad([28,28], [32,32], 1, pad=True, pad_value=-1),
                        c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                        decision_dist_fn='eucos',
                        dropout = True, keep_prob=0.2,
                        batch_size=256,
                        iterations=1500,
                        display_step=-500,
                        tailsize = 20,
                        alpharank = 3),
                y_dim=6, batch_size=128,
                unpad_flatten=flatten_unpad([32,32], [28,28],1))

    elif dataset_name == 'cifar10':
            if model_name == 'recon-self-supervision':
                return lambda: OpenNetCNN(
                    [36, 36],  # x_dim
                    1,  # x_chs
                    6,  # y_dim
                    [32, 64],  # conv_units,
                    [256, 128],  # hidden_units
                    z_dim=6 if z_dim is None else z_dim,
                    kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                    pooling_enable=True, pooling_kernel=[3, 3],
                    pooling_stride=[2, 2], pooling_padding='SAME',
                    pooling_type='max',
                    activation_fn=tf.nn.relu,

                    x_scale=UnitPosNegScale.scale,
                    x_inverse_scale=UnitPosNegScale.inverse_scale,
                    x_reshape=reshape_pad([32, 32], [36, 36], 1, pad=True, pad_value=-1),

                    opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                    recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                    c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),

                    dist='triplet_center_loss',
                    decision_dist_fn='mahalanobis',
                    dropout=True, keep_prob=0.2,
                    batch_size=256,
                    iterations=8000,
                    display_step=-500,
                    save_step=1000,
                    model_directory=model_directory,  # Directory to save trained model to.
                    density_estimation_factory=None,
                    ce_loss=False, recon_loss=True, inter_loss=False, intra_loss=False, div_loss=False,
                    mmf_extension=False, tc_loss=False, contamination=0.01, margin=2.0, n_rots=[36,36], self_sup = False
                )
            elif model_name == 'self-supervision':
                return lambda: OpenNetCNN(
                    [36, 36],  # x_dim
                    1,  # x_chs
                    6,  # y_dim
                    [32, 64],  # conv_units,
                    [256, 128],  # hidden_units
                    z_dim=6 if z_dim is None else z_dim,
                    kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                    pooling_enable=True, pooling_kernel=[3, 3],
                    pooling_stride=[2, 2], pooling_padding='SAME',
                    pooling_type='max',
                    activation_fn=tf.nn.relu,

                    x_scale=UnitPosNegScale.scale,
                    x_inverse_scale=UnitPosNegScale.inverse_scale,
                    x_reshape=reshape_pad([32, 32], [36, 36], 1, pad=True, pad_value=-1),

                    opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                    recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                    c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),

                    dist='triplet_center_loss',
                    decision_dist_fn='mahalanobis',
                    dropout=True, keep_prob=0.2,
                    batch_size=256,
                    iterations=8000,
                    display_step=-500,
                    save_step=1000,
                    model_directory=model_directory,  # Directory to save trained model to.
                    density_estimation_factory=None,
                    ce_loss=False, recon_loss=False, inter_loss=False, intra_loss=False, div_loss=False,
                    mmf_extension=False, tc_loss=False,
                    contamination=0.01, margin=2.0, n_rots=4, self_sup = True
                )
            elif model_name == 'triplet':
                return lambda: OpenNetCNN(
                    [36, 36],  # x_dim
                    1,  # x_chs
                    6,  # y_dim
                    [32, 64],  # conv_units,
                    [256, 128],  # hidden_units
                    z_dim=6 if z_dim is None else z_dim,
                    kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                    pooling_enable=True, pooling_kernel=[3, 3],
                    pooling_stride=[2, 2], pooling_padding='SAME',
                    pooling_type='max',
                    activation_fn=tf.nn.relu,

                    x_scale=UnitPosNegScale.scale,
                    x_inverse_scale=UnitPosNegScale.inverse_scale,
                    x_reshape=reshape_pad([32, 32], [36, 36], 1, pad=True, pad_value=-1),

                    opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                    recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                    c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),

                    dist='triplet_center_loss',
                    decision_dist_fn='mahalanobis',
                    dropout=True, keep_prob=0.2,
                    batch_size=256,
                    iterations=5000,
                    display_step=-500,
                    save_step=500,
                    model_directory=model_directory,  # Directory to save trained model to.
                    load_model_directory=load_model_directory,
                    density_estimation_factory=None,
                    ce_loss=False, recon_loss=False, inter_loss=False, intra_loss=False, div_loss=False,
                    mmf_extension=False, tc_loss=True,
                    contamination=0.01, margin=2.0
                )

            elif model_name == 'ii':
                return lambda: OpenNetCNN(
                    [36, 36],  # x_dim
                    1,  # x_ch
                    6,  # y_dim
                    [32, 64],  # conv_units,
                    [256, 128],  # hidden_units
                    z_dim=6 if z_dim is None else z_dim,
                    kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                    pooling_enable=True, pooling_kernel=[3, 3],
                    pooling_stride=[2, 2], pooling_padding='SAME',
                    pooling_type='max',
                    activation_fn=tf.nn.relu,

                    x_scale=UnitPosNegScale.scale,
                    x_inverse_scale=UnitPosNegScale.inverse_scale,
                    x_reshape=reshape_pad([32, 32], [36, 36], 1, pad=True, pad_value=-1),

                    opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                    recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                    c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),

                    dist='mean_separation_spread',
                    decision_dist_fn='mahalanobis',
                    dropout=True, keep_prob=0.2,
                    batch_size=256,
                    iterations=5000,
                    display_step=-500,
                    save_step=500,
                    model_directory=model_directory,  # Directory to save trained model to.
                    load_model_directory=load_model_directory,
                    density_estimation_factory=None,
                    ce_loss=False, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False,
                    mmf_extension=False,
                    contamination=0.01
                )
            elif model_name == 'ce':
                return lambda: OpenNetCNN(
                    [36, 36],  # x_dim
                    1,  # x_ch
                    6,  # y_dim
                    [32, 64],  # conv_units,
                    [256, 128],  # hidden_units
                    z_dim=6 if z_dim is None else z_dim,
                    kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                    pooling_enable=True, pooling_kernel=[3, 3],
                    pooling_stride=[2, 2], pooling_padding='SAME',
                    pooling_type='max',
                    activation_fn=tf.nn.relu,

                    x_scale=UnitPosNegScale.scale,
                    x_inverse_scale=UnitPosNegScale.inverse_scale,
                    x_reshape=reshape_pad([32, 32], [36, 36], 1, pad=True, pad_value=-1),

                    opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                    recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                    c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),

                    dist='mean_separation_spread',
                    decision_dist_fn='mahalanobis',
                    dropout=True, keep_prob=0.2,
                    batch_size=256,
                    iterations=5000,
                    display_step=-500,
                    save_step=500,
                    model_directory=model_directory,  # Directory to save trained model to.
                    load_model_directory=load_model_directory,
                    density_estimation_factory=None,
                    ce_loss=True, recon_loss=False, inter_loss=False, intra_loss=False, div_loss=False,
                    contamination=0.01,
                )
            elif model_name == 'ceii':
                return lambda: OpenNetCNN(
                    [36, 36],  # x_dim
                    1,  # x_ch
                    6,  # y_dim
                    [32, 64],  # conv_units,
                    [256, 128],  # hidden_units
                    z_dim=6 if z_dim is None else z_dim,
                    kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                    pooling_enable=True, pooling_kernel=[3, 3],
                    pooling_stride=[2, 2], pooling_padding='SAME',
                    pooling_type='max',
                    activation_fn=tf.nn.relu,

                    x_scale=UnitPosNegScale.scale,
                    x_inverse_scale=UnitPosNegScale.inverse_scale,
                    x_reshape=reshape_pad([32, 32], [36, 36], 1, pad=True, pad_value=-1),

                    opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                    recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                    c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),

                    dist='mean_separation_spread',
                    decision_dist_fn='mahalanobis',
                    dropout=True, keep_prob=0.2,
                    batch_size=256,
                    iterations=8000,
                    display_step=-500,
                    save_step=500,
                    model_directory=None,  # Directory to save trained model to.
                    density_estimation_factory=None,
                    ce_loss=True, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False,
                    contamination=0.01,
                )
            elif model_name == 'central':
                return lambda: CentralOpennetCNN(
                    [36, 36],  # x_dim
                    1,  # x_ch
                    6,  # y_dim
                    [32, 64],  # conv_units,
                    [256, 128],  # hidden_units
                    kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                    pooling_enable=True, pooling_kernel=[3, 3],
                    pooling_stride=[2, 2], pooling_padding='SAME',
                    pooling_type='max',
                    activation_fn=tf.nn.relu,

                    x_scale=UnitPosNegScale.scale,
                    x_inverse_scale=UnitPosNegScale.inverse_scale,
                    x_reshape=reshape_pad([32, 32], [36, 36], 1, pad=True, pad_value=-1),

                    opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),

                    decision_dist_fn='mahalanobis',
                    dropout=True, keep_prob=0.2,
                    batch_size=256,
                    iterations=5000,
                    display_step=-500,
                    save_step=500,
                    model_directory=None,  # Directory to save trained model to.
                    contamination=0.01,
                    penalty=0.01,
                )
            elif model_name == 'openmax':
                return lambda: OpenMaxCNN(
                    [36, 36],  # x_dim
                    1,  # x_ch
                    6,  # y_dim
                    [32, 64],  # conv_units,
                    [256, 128],  # hidden_units
                    kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                    pooling_enable=True, pooling_kernel=[3, 3],
                    pooling_stride=[2, 2], pooling_padding='SAME',
                    pooling_type='max',
                    activation_fn=tf.nn.relu,

                    x_scale=UnitPosNegScale.scale,
                    x_inverse_scale=UnitPosNegScale.inverse_scale,
                    x_reshape=reshape_pad([32, 32], [36, 36], 1, pad=True, pad_value=-1),

                    c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),

                    decision_dist_fn='eucos',
                    dropout=True, keep_prob=0.2,
                    batch_size=256,
                    iterations=5000,
                    display_step=-500,
                    save_step=500,
                    model_directory=None,  # Directory to save trained model to.
                    tailsize=20,
                    alpharank=3,
                )
            elif model_name == 'g_openmax':
                return lambda: GOpenmax(
                    gan_factory=lambda y_dim: MultiClassGAN(
                        [36, 36],  # x_dim
                        1,  # x_ch
                        y_dim,  # y_dim
                        z_dim=100,
                        generator=GeneratorDC([32, 32],  # x_dims
                                              1,  # x_ch
                                              [64, 32, 16],  # g_conv_units
                                              g_kernel_sizes=[5, 5], g_strides=[2, 2], g_paddings='SAME',
                                              g_activation_fn=tf.nn.relu),  # Generator Net
                        discriminator=DiscriminatorDC(6,  # y_dim
                                                      [16, 32, 64],  # conv_units
                                                      hidden_units=None,
                                                      kernel_sizes=[5, 5], strides=[2, 2], paddings='SAME',
                                                      d_activation_fn=tf.contrib.keras.layers.LeakyReLU,
                                                      f_activation_fns=tf.nn.relu,
                                                      dropout=False, keep_prob=0.5),  # Discriminator Net
                        x_reshape=reshape_pad([28, 28], [32, 32], 1, pad=True, pad_value=-1),
                        x_scale=UnitPosNegScale.scale,
                        x_inverse_scale=UnitPosNegScale.inverse_scale,
                        d_optimizer=tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5),
                        g_optimizer=tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5),
                        g_loss_fn='default',
                        d_label_smooth=0.75,
                        ## Training config
                        batch_size=128,
                        iterations=5000,
                        display_step=-500,
                        save_step=500,
                    ),
                    openmax_factory=lambda y_dim: OpenMaxCNN(
                        [32, 32], 1,  # x_ch
                        y_dim,  # y_dim
                        [32, 64],  # conv_units,
                        [256, 128],  # hidden_units
                        kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                        pooling_enable=True, pooling_kernel=[3, 3],
                        pooling_stride=[2, 2], pooling_padding='SAME',
                        pooling_type='max',
                        activation_fn=tf.nn.relu,
                        x_scale=UnitPosNegScale.scale,
                        x_inverse_scale=UnitPosNegScale.inverse_scale,
                        x_reshape=reshape_pad([28, 28], [32, 32], 1, pad=True, pad_value=-1),
                        c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                        decision_dist_fn='eucos',
                        dropout=True, keep_prob=0.2,
                        batch_size=256,
                        iterations=1500,
                        display_step=-500,
                        save_step=500,
                        tailsize=20,
                        alpharank=3),
                    classifier_factory=lambda y_dim: OpenMaxCNN(
                        [32, 32], 1,  # x_ch
                        y_dim,  # y_dim
                        [32, 64],  # conv_units,
                        [256, 128],  # hidden_units
                        kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                        pooling_enable=True, pooling_kernel=[3, 3],
                        pooling_stride=[2, 2], pooling_padding='SAME',
                        pooling_type='max',
                        activation_fn=tf.nn.relu,
                        x_scale=UnitPosNegScale.scale,
                        x_inverse_scale=UnitPosNegScale.inverse_scale,
                        x_reshape=reshape_pad([28, 28], [32, 32], 1, pad=True, pad_value=-1),
                        c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                        decision_dist_fn='eucos',
                        dropout=True, keep_prob=0.2,
                        batch_size=256,
                        iterations=1500,
                        display_step=-500,
                        tailsize=20,
                        alpharank=3),
                    y_dim=6, batch_size=128,
                    unpad_flatten=flatten_unpad([32, 32], [28, 28], 1))


def get_closed_cnn_model_factories(model_name, dataset_name, open_dataset, z_dim=None):

    if dataset_name == 'mnist':
        if model_name == 'mmf':
            return lambda: OpenNetCNN(
                [32, 32],  # x_dim
                1,  # x_ch
                10,  # y_dim
                [32, 64],  # conv_units,
                [256, 128],  # hidden_units
                z_dim=10 if z_dim is None else z_dim,
                kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                pooling_enable=True, pooling_kernel=[3, 3],
                pooling_stride=[2, 2], pooling_padding='SAME',
                pooling_type='max',
                activation_fn=tf.nn.relu,

                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                x_reshape=reshape_pad([28, 28], [32, 32], 1, pad=True, pad_value=-1),

                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),

                dist='mean_separation_spread',
                decision_dist_fn='mahalanobis',
                dropout=True, keep_prob=0.2,
                batch_size=256,
                iterations=5000,
                display_step=-500,
                save_step=500,
                model_directory=None,  # Directory to save trained model to.
                density_estimation_factory=None,
                ce_loss=True, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False, mmf_extension=True,
                contamination=0.01, mmf_comb=0.1
            )
        elif model_name == 'ii':
            return lambda : OpenNetCNN(
                [32, 32],  # x_dim
                1,  #x_ch
                10,  #y_dim
                [32, 64], # conv_units,
                [256, 128],      #hidden_units
                z_dim=10 if z_dim is None else z_dim,
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
                decision_dist_fn='mahalanobis',
                dropout = True, keep_prob=0.2,
                batch_size=256,
                iterations=5000,
                display_step=-500,
                save_step=500,
                model_directory=None,  # Directory to save trained model to.
                density_estimation_factory=None,
                ce_loss=False, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False,
                contamination=0.01,
            )
        elif model_name == 'ceii':
            return lambda : OpenNetCNN(
                [32, 32],  # x_dim
                1,  #x_ch
                10,  #y_dim
                [32, 64], # conv_units,
                [256, 128],      #hidden_units
                z_dim=10 if z_dim is None else z_dim,
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
                decision_dist_fn='mahalanobis',
                dropout = True, keep_prob=0.2,
                batch_size=256,
                iterations=5000,
                display_step=-500,
                save_step=500,
                model_directory=None,  # Directory to save trained model to.
                density_estimation_factory=None,
                ce_loss=True, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False,
                contamination=0.01,
            )
        elif model_name == 'openmax':
            return lambda : OpenMaxCNN(
                [32, 32],  # x_dim
                1,  #x_ch
                10,  #y_dim
                [32, 64], # conv_units,
                [256, 128],      #hidden_units
                kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                pooling_enable=True, pooling_kernel=[3,3],
                pooling_stride=[2,2], pooling_padding='SAME',
                pooling_type='max',
                activation_fn=tf.nn.relu,

                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                x_reshape=reshape_pad([28,28], [32,32], 1, pad=True, pad_value=-1),

                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),

                decision_dist_fn='eucos',
                dropout = True, keep_prob=0.2,
                batch_size=256,
                iterations=5000,
                display_step=-500,
                save_step=500,
                model_directory=None,  # Directory to save trained model to.
                tailsize = 20,
                alpharank = 4,
            )


def single_exp(exp_id, network_type, model_name, model_factory, dataset_name, dataset, output_dir, tr_classes, pre_model=None, transformation=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    if transformation != 'none':
        n_dim = None
        p = pre_model()
        if dataset_name == "mnist" or dataset_name == "fashion-mnist":
            n_dim = 28
        elif dataset_name == "msadjmat":
            n_dim = 63
        elif dataset_name == "cifar10":
            n_dim = 32
        elif dataset_name == "android":
            n_dim = 1453
        elif dataset_name == "stl10":
            n_dim = 96
        print("Transformation type = ", transformation)
        if transformation == "affine":
            x_train_trans, y_train_trans, trans = affine_transformation(dataset.train_data(), dataset.train_label(), n_dim, 3)
        elif transformation == "crop":
            x_train_trans, y_train_trans, trans = crop_transformation(dataset.train_data(), dataset.train_label(), n_dim, 3)
        elif transformation == "random":
            x_train_trans, y_train_trans, trans = random_order_transformation(dataset.train_data(), dataset.train_label(), n_dim, 3)
        elif transformation == "ae-random":
            x_train_trans, y_train_trans, trans = ae_random_order_transformation(dataset.train_data(), dataset.train_label(), n_dim, 3)
        elif transformation == "ae-shift":
            x_train_trans, y_train_trans, trans = ae_shift_transformation(dataset.train_data(), dataset.train_label(), n_dim, 3)
        elif transformation == "shift":
            x_train_trans, y_train_trans, trans = shift_transformation(dataset.train_data(), dataset.train_label(), n_dim, 3)
        elif transformation == "ae-swap":
            x_train_trans, y_train_trans, trans = ae_swap_transformation(dataset.train_data(), dataset.train_label(), n_dim, 3)
        elif transformation == "offset":
            x_train_trans, y_train_trans, trans = offset_transformation(dataset.train_data(), dataset.train_label(), n_dim, 3)
        elif transformation == "misc":
            x_train_trans, y_train_trans, trans = ae_img_misc_transformation(dataset.train_data(), dataset.train_label(), n_dim)
        elif transformation == "gaussian":
            x_train_trans, y_train_trans, trans = gaussian_transformation(dataset.train_data(), dataset.train_label(), n_dim, 3)
        elif transformation == "ae-gaussian":
            x_train_trans, y_train_trans, trans = ae_gaussian_transformation(dataset.train_data(), dataset.train_label(), n_dim, 3)
        elif transformation == "ae-affine":
            x_train_trans, y_train_trans, trans = ae_affine_transformation(dataset.train_data(), dataset.train_label(), n_dim, 3)
        elif transformation == "rotation":
            tr_class_str = ''.join([str(i) for i in tr_classes])
            if not os.path.exists(output_dir+"/rotation/"):
                os.makedirs(output_dir+"/rotation/")
            if not os.path.exists(output_dir+"/rotation/"+tr_class_str+"_transX.npy"):
                x_train_trans, y_train_trans, trans = rotate_transformation(dataset.train_data(), dataset.train_label(), n_dim, 3)
                with open(output_dir+"/rotation/"+tr_class_str+"_transX.npy", 'wb') as transx:
                    np.save(transx, x_train_trans)
                with open(output_dir+"/rotation/"+tr_class_str+"_transY.npy", 'wb') as transy:
                    np.save(transy, y_train_trans)
                with open(output_dir+"/rotation/"+tr_class_str+"_trans.npy", 'wb') as t:
                    np.save(t, trans)
            else:
                with open(output_dir+"/rotation/"+tr_class_str+"_transX.npy", 'rb') as transx:
                    x_train_trans = np.load(transx)
                with open(output_dir+"/rotation/"+tr_class_str+"_transY.npy", 'rb') as transy:
                    y_train_trans = np.load(transy)
                with open(output_dir+"/rotation/"+tr_class_str+"_trans.npy", 'rb') as t:
                    trans = np.load(t)
        elif transformation == "ae-rotation":
            tr_class_str = ''.join([str(i) for i in tr_classes])
            if not os.path.exists(output_dir+"/ae-rotation/"):
                os.makedirs(output_dir+"/ae-rotation/")
            if not os.path.exists(output_dir+"/ae-rotation/"+tr_class_str+"_transX.npy"):
                x_train_trans, y_train_trans, trans = ae_rotate_transformation(dataset.train_data(), dataset.train_label(), n_dim, 3)
                with open(output_dir+"/ae-rotation/"+tr_class_str+"_transX.npy", 'wb') as transx:
                    np.save(transx, x_train_trans)
                with open(output_dir+"/ae-rotation/"+tr_class_str+"_transY.npy", 'wb') as transy:
                    np.save(transy, y_train_trans)
                with open(output_dir+"/ae-rotation/"+tr_class_str+"_trans.npy", 'wb') as t:
                    np.save(t, trans)
            else:
                with open(output_dir+"/ae-rotation/"+tr_class_str+"_transX.npy", 'rb') as transx:
                    x_train_trans = np.load(transx)
                with open(output_dir+"/ae-rotation/"+tr_class_str+"_transY.npy", 'rb') as transy:
                    y_train_trans = np.load(transy)
                with open(output_dir+"/ae-rotation/"+tr_class_str+"_trans.npy", 'rb') as t:
                    trans = np.load(t)
        elif transformation == "shift-ae-rotation":
            tr_class_str = ''.join([str(i) for i in tr_classes])
            if not os.path.exists(output_dir+"/shift-ae-rotation/"):
                os.makedirs(output_dir+"/shift-ae-rotation/")
            if not os.path.exists(output_dir+"/shift-ae-rotation/"+tr_class_str+"_transX.npy"):
                x_train_trans, y_train_trans, trans = ae_shift_rotate_transformation(dataset.train_data(), dataset.train_label(), n_dim, 4)
                with open(output_dir+"/shift-ae-rotation/"+tr_class_str+"_transX.npy", 'wb') as transx:
                    np.save(transx, x_train_trans)
                with open(output_dir+"/shift-ae-rotation/"+tr_class_str+"_transY.npy", 'wb') as transy:
                    np.save(transy, y_train_trans)
                with open(output_dir+"/shift-ae-rotation/"+tr_class_str+"_trans.npy", 'wb') as t:
                    np.save(t, trans)
            else:
                with open(output_dir+"/shift-ae-rotation/"+tr_class_str+"_transX.npy", 'rb') as transx:
                    x_train_trans = np.load(transx)
                with open(output_dir+"/shift-ae-rotation/"+tr_class_str+"_transY.npy", 'rb') as transy:
                    y_train_trans = np.load(transy)
                with open(output_dir+"/shift-ae-rotation/"+tr_class_str+"_trans.npy", 'rb') as t:
                    trans = np.load(t)
       # print("start pre-training")
        p.fit(x_train_trans, y_train_trans, rot=trans)

    elif pre_model:
        p = pre_model()
        print("start pre-training")
        p.fit(dataset.train_data(), dataset.train_label())

    m = model_factory()
   # print("Transformation type = ", transformation)
  #  if transformation == "random":
   #     x_train_trans, y_train_trans, trans = random_order_transformation(dataset.train_data(), dataset.train_label(), 63, 3)

    print('m.y_dim=', m.y_dim)
    train_start = time.time()
    print("start time: ", train_start)
    print(dataset.train_data().shape, dataset.train_label().shape)
    m.fit(dataset.train_data(), dataset.train_label())
    train_end = time.time()
    print("end time: ", train_end)
    print("training time: ", int(train_end-train_start))


    result = {}
    result['dataset_name'] = dataset_name
    result['model_name'] = model_name
    result['network_type'] = network_type
    result['tr_classes'] = tr_classes
    try:
        result['model_config'] = m.model_config()
        result['class_mean'] = m.c_means
        result['class_cov'] = m.c_cov
        result['class_cov_inv'] = m.c_cov_inv
    except:
        pass
    result['train_decision_function'] = m.decision_function(dataset.train_data())
    result['test_decision_function'] = m.decision_function(dataset.test_data())
    try:
        result['train_dist_all_class'] = m.distance_from_all_classes(dataset.train_data())
        result['test_dist_all_class'] = m.distance_from_all_classes(dataset.test_data())
    except:
        pass
    try:
        result['test_predict_prob'] = m.predict_prob(dataset.test_data())
    except:
        pass
    try:
        result['test_predict_prob_open'] = m.predict_prob_open(dataset.test_data())
    except:
        pass
    try:
        result['test_closed_predict_y'] = m.predict(dataset.test_data())
    except:
        pass
    try:
        result['test_open_predict_y'] = m.predict_open(dataset.test_data())[0]
    except:
        pass
    try:
        result['train_z'] = m.latent(dataset.train_data())
        result['test_z'] = m.latent(dataset.test_data())
    except:
        pass
    result['train_true_y'] = dataset.train_label()
    result['test_true_y'] = dataset.test_label()
    result['train_time'] = int(train_end - train_start)

    fmt='{dataset_name}_{network_type}_{model_name}_%Y_%m_%d_%H_%M_%S_e{exp_id}.pkl.gz'
    file_name = datetime.datetime.now().strftime(fmt).format(
        dataset_name=dataset_name, network_type=network_type, model_name=model_name, exp_id=exp_id)
    save_pickle_gz(result, os.path.join(output_dir, file_name))

    time.sleep(30)

    return m

def main():
    parser = argparse.ArgumentParser(description='OpenNetFlat experiments.')
    parser.add_argument('-eid', '--exp_id', required=True, dest='exp_id',
                        help='path to output directory.')
    parser.add_argument('-ds','--dataset', required=True, dest='dataset_name',
                        choices=['mnist', 'fashion-mnist', 'cifar10'], help='dataset name.')
    parser.add_argument('-n','--network', required=True, dest='network',
                        choices=['flat', 'cnn'], help='dataset name.')
    parser.add_argument('-m','--model', required=True, dest='model_name',
                        choices=['ii', 'ce', 'ceii', 'openmax', 'g_openmax', 'central','triplet'], help='model name.')
    parser.add_argument('-trc', '--tr_classes', required=True, dest='tr_classes', nargs='+',
                        type=int, help='list of training classes.')
    parser.add_argument('-o', '--outdir', required=False, dest='output_dir',
                        default='./exp_result/cnn', help='path to output directory.')
    parser.add_argument('-z', '--zdim', required=False, dest='z_dim', type=int,
                        default=None, help='[optional] dimension of z layer.')
    parser.add_argument('-s', '--seed', required=False, dest='seed', type=int,
                        default=1, help='path to output directory.')
    parser.add_argument('--closed', dest='closed', action='store_true')
    parser.add_argument('--no-closed', dest='closed', action='store_false')
    parser.add_argument('-p','--pre-trained', required=False, default='false',  choices=['false', 'trans', 'recon'], dest='pre_trained', help='Use self-supervision pre-trained model: True/False')
    parser.add_argument('-t','--transformation', required=False, dest='transformation', choices=['none', 'shift', 'random', 'misc', 'random1d', 'rotation', 'affine', 'crop', 'gaussian', 'offset', 'ae-gaussian', 'ae-swap', 'ae-shift', 'ae-affine','ae-rotation','ae-random','shift-ae-rotation'], help='Tranformation type')
    parser.set_defaults(closed=False)

    args = parser.parse_args()
    print ("Loading Dataset: " + args.dataset_name)
    open_dataset = load_open_dataset(args.dataset_name, args.tr_classes, args.seed)
    print ("Creating Model Config: " + args.network)
    pre_trained_model = None
    if args.network == 'flat':
        if args.closed:
            model_factory = get_closed_flat_model_factories(
                args.model_name, args.dataset_name, open_dataset, z_dim=args.z_dim)
        else:
            if args.pre_trained != "false":
                pretrained_model_directory = "model/"+args.dataset_name+"/"+args.network+"/"+args.model_name+"/pretrain/"+args.transformation
                load_pretrained_model_directory = pretrained_model_directory + "/model-8000.cptk"
                if args.pre_trained == "trans":
                    pre_trained_model = get_flat_model_factories("self-supervision", args.dataset_name, open_dataset, z_dim=args.z_dim, model_directory=pretrained_model_directory)
                elif args.pre_trained == "recon":
                    pre_trained_model = get_flat_model_factories("recon-self-supervision", args.dataset_name, open_dataset, z_dim=args.z_dim, model_directory=pretrained_model_directory)
                model_factory = get_flat_model_factories(args.model_name, args.dataset_name, open_dataset, z_dim=args.z_dim, load_model_directory=load_pretrained_model_directory)
            else:
                model_factory = get_flat_model_factories(args.model_name, args.dataset_name, open_dataset, z_dim=args.z_dim)
    elif args.network == 'cnn':
        if args.closed:
            model_factory = get_closed_cnn_model_factories(
                args.model_name, args.dataset_name, open_dataset, z_dim=args.z_dim)
        else:
            if args.pre_trained != "false":
                pretrained_model_directory = "model/"+args.dataset_name+"/"+args.network+"/"+args.model_name+"/pretrain/"+args.transformation
                load_pretrained_model_directory = pretrained_model_directory + "/model-6000.cptk"
                if args.pre_trained == "trans":
                    pre_trained_model = get_cnn_model_factories("self-supervision", args.dataset_name, open_dataset, z_dim=args.z_dim, model_directory=pretrained_model_directory)
                elif args.pre_trained == "recon":
                    pre_trained_model = get_cnn_model_factories("recon-self-supervision", args.dataset_name, open_dataset, z_dim=args.z_dim, model_directory=pretrained_model_directory)
                model_factory = get_cnn_model_factories(args.model_name, args.dataset_name, open_dataset, z_dim=args.z_dim, load_model_directory=load_pretrained_model_directory)
            else:
                model_factory = get_cnn_model_factories(args.model_name, args.dataset_name, open_dataset, z_dim=args.z_dim)

    print ("Starting single experiment...")
    single_exp(args.exp_id, args.network, args.model_name, model_factory,
               args.dataset_name, open_dataset, args.output_dir, args.tr_classes, pre_model=pre_trained_model, transformation=args.transformation)
    print ("Finished single experiment...")


if __name__ == '__main__':
    main()
