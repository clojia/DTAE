import argparse
import subprocess
import random
import os
import tensorflow as tf
import sys
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
from tensorflow.python.client import device_lib



def main():
    parser = argparse.ArgumentParser(description='OpenNetFlat experiments.')
    parser.add_argument('-eid', '--exp_id', required=False, dest='exp_id',
                        default=None, help='path to output directory.')
    parser.add_argument('-n','--network', required=True, dest='network',
                        choices=['flat', 'cnn'], help='dataset name.')
    parser.add_argument('-ds','--datasets', required=True, dest='dataset_name',
                        choices=['mnist', 'fashion-mnist', 'cifar10'], help='dataset name.')
    parser.add_argument('-m','--models', required=True, dest='model_names', nargs='+',
                        default=['ii', 'ce', 'ceii', 'openmax', 'g_openmax', 'central','triplet',], help='model name.')
    parser.add_argument('-trc_file', '--tr_classes_list_file', required=True, dest='trc_file',
                        help='list of training classes.')
    parser.add_argument('-o', '--outdir', required=False, dest='output_dir',
                        default='./exp_result/cnn', help='path to output directory.')
    parser.add_argument('-s', '--seed', required=False, dest='seed', type=int,
                        default=1, help='path to output directory.')
    parser.add_argument('--closed', dest='closed', action='store_true',
                        help='Run closed world experiments.')
    parser.add_argument('--no-closed', dest='closed', action='store_false',
                        help='Run open world experiments.')
    parser.add_argument('-p','--pre-trained', required=False, default='false',  dest='pre_trained',  choices=['false','recon','trans'], help='Use self-supervision pre-trained model: True/False')
    parser.add_argument('-t','--transformation', required=False, dest='transformation', choices=['none','random','shift','ae-shift','ae-swap','random1d','ae-affine','ae-gaussian','ae-rotation','shift-ae-rotation','ae-random','rotation','affine', 'crop', 'gaussian', 'offset', 'misc'], help='Tranformation type')
    parser.set_defaults(closed=False)

    args = parser.parse_args()

    if args.exp_id is None:
        args.exp_id = random.randint(0, 10000)

    tr_classes_list = []
    with open(args.trc_file) as fin:
        for line in fin:
            if line.strip() == '':
                continue
            cols = line.strip().split()
            tr_classes_list.append([int(float(c)) for c in cols])

    for tr_classes in tr_classes_list:
        for mname in args.model_names:
            exp_args = []
            exp_args += ['python', 'exp_opennet.py']
            exp_args += ['-e', str(args.exp_id)]
            exp_args += ['-n', args.network]
            exp_args += ['-m', mname]
            exp_args += ['-ds', args.dataset_name]
            exp_args += ['-trc']
            exp_args += [str(c) for c in tr_classes[:10]]
            exp_args += ['-o', args.output_dir]
            exp_args += ['-s', str(args.seed)]
            exp_args += ['-p', str(args.pre_trained)]
            exp_args += ['--transformation', args.transformation]

            if args.closed:
                exp_args += ['--closed']

            print(exp_args)
            proc = subprocess.Popen(exp_args)
            proc.wait()

if __name__ == '__main__':
    main()
