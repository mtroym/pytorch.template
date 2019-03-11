import argparse
import os
import random

import torch
import torch.backends.cudnn as cudnn


def parse():
    parser = argparse.ArgumentParser()
    # General options
    parser.add_argument('--logNum', default=100, type=int, help='Log interval')
    parser.add_argument('--debug', default=False, type=str2bool, help='Debug mode')
    parser.add_argument('--manualSeed', default=0, type=int, help='manual seed')
    parser.add_argument('--GPU', default=True, type=str2bool, help='Use GPU')
    parser.add_argument('--GPUs', default='1', type=str, help='ID of GPUs to use, seperate by ,')
    parser.add_argument('--backend', default='cudnn', type=str, help='backend', choices=['cudnn', 'cunn'])
    parser.add_argument('--cudnn', default='fastest', type=str, help='cudnn setting',
                        choices=['fastest', 'deterministic', 'default'])
    # Path options
    parser.add_argument('--data', default='../data', type=str, help='Path to dataset')
    parser.add_argument('--gen', default='../gen', type=str, help='Path to generated files')
    parser.add_argument('--resume', default='../models', type=str, help='Path to checkpoint')
    parser.add_argument('--www', default='../www', type=str, help='Path to visualization')
    # Data options
    parser.add_argument('--dataset', default='CelebA', type=str, help='Name of dataset')
    parser.add_argument('--nThreads', default=8, type=int, help='Number of data loading threads')
    parser.add_argument('--trainPctg', default=0.95, type=float, help='Percentage of training images')
    parser.add_argument('--imgDim', default=224, type=int, help='Image dimension')
    # Training/testing options
    parser.add_argument('--logDir', default='./log_dir', type=str, help='Tensorboard dir.')
    parser.add_argument('--nEpochs', default=100, type=int, help='Number of total epochs to run')
    parser.add_argument('--metrics', default='[PSNR, SSIM]', type=str2list, help='metrics in eval part')
    parser.add_argument('--epochNum', default=-1, type=int, help='0=retrain | -1=latest | -2=best', choices=[0, -1, -2])
    parser.add_argument('--batchSize', default=64, type=int, help='mini-batch size')
    parser.add_argument('--saveEpoch', default=5, type=int, help='saving at least # epochs')
    parser.add_argument('--testOnly', default=False, type=str2bool, help='Run the test to see the performance')
    parser.add_argument('--barwidth', default=40, type=int, help='Progress bar width')
    parser.add_argument('--visTrain', default=2, type=int, help='Visualizing training examples')
    parser.add_argument('--visTest', default=2, type=int, help='Visualizing testing examples')
    parser.add_argument('--visWidth', default=1, type=int, help='Number of images per row for visualization')
    parser.add_argument('--visThres', default=0.2, type=float, help='Threshold for visualization')
    # Optimization options
    parser.add_argument('--LR', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--LRDecay', default='exp', type=str, help='LRDecay method')
    parser.add_argument('--LRDParam', default=10, type=int, help='param for learning rate decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--dampening', default=0.0, type=float, help='dampening')
    parser.add_argument('--weightDecay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--optimizer', default='SGD', type=str, help='optimizertype, more choices available',
                        choices=['SGD', 'Adam'])
    # Model options
    parser.add_argument('--netType', default='srcnn', type=str, help='Your defined model name')
    parser.add_argument('--netSpec', default='custom', type=str, help='Other model to be loaded',
                        choices=['custom', 'resnet'])
    parser.add_argument('--pretrain', default=False, type=str2bool, help='Pretrained or not')
    parser.add_argument('--absLoss', default=0, type=float, help='Weight for abs criterion')
    parser.add_argument('--bceLoss', default=0, type=float, help='Weight for bce criterion')
    parser.add_argument('--mseLoss', default=1, type=float, help='Weight for mse criterion')
    parser.add_argument('--frozen', default=False, type=str2bool, help='Weather freeze the pretrain')
    parser.add_argument('--backbone', default='Resnet', type=str, help='Other model to be loaded', choices=['Resnet', 'Xception'])

    # Other model options
    parser.add_argument('--numClasses', default=5, type=int, help='Number of classes in the dataset')
    parser.add_argument('--suffix', default='', type=str, help='Suffix for saving the model')
    parser.add_argument('--dropoutRate', default=0.2, type=float, help='Drop out Rate of fc.')
    parser.add_argument('--numChannels', default=64, type=int, help='number of channel of srcnn')
    opt = parser.parse_args()

    opt.GPU = opt.GPU & torch.cuda.is_available()
    if opt.GPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.GPUs
        cudnn.benchmark = True

    torch.set_default_tensor_type('torch.FloatTensor')

    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.GPU:
        torch.cuda.manual_seed_all(opt.manualSeed)

    # for arnold script.
    print('**************** DEFINE SOME PATH *****************')
    print('\t-=> opt.data='+opt.data)
    print('\t-=> opt.gen='+opt.gen)
    print('\t-=> opt.www='+opt.www)
    print('\t-=> opt.resume='+opt.resume)
    print('**************** ^^^^^^^^^^^^^^^^ *****************')

    if opt.debug:
        opt.nEpochs = 1
        opt.nThreads = min(opt.nThreads, 1)
        opt.visTrain = min(opt.visTrain, 10)
        opt.visTest = min(opt.visTest, 10)

    opt.hashKey = opt.dataset + '_' + opt.netType
    if opt.pretrain:
        opt.hashKey = opt.hashKey + '_pre'
    if opt.absLoss != 0:
        opt.hashKey = opt.hashKey + '_abs' + str(opt.absLoss)
    if opt.mseLoss != 0:
        opt.hashKey = opt.hashKey + '_mse' + str(opt.mseLoss)
    if opt.bceLoss != 0:
        opt.hashKey = opt.hashKey + '_bce' + str(opt.bceLoss)
    opt.hashKey = opt.hashKey + '_LR' + str(opt.LR)
    if opt.suffix != '':
        opt.hashKey = opt.hashKey + '_' + str(opt.suffix)

    opt.dataRoot = opt.data
    opt.data = os.path.join(opt.data, opt.dataset)
    opt.gen = os.path.join(opt.gen, opt.dataset)
    opt.resume = os.path.join(opt.resume, opt.hashKey)
    opt.www = os.path.join(opt.www, opt.hashKey)

    if not os.path.exists(opt.gen):
        os.makedirs(opt.gen)
    if not os.path.exists(opt.resume):
        os.makedirs(opt.resume)
    if not os.path.exists(opt.www):
        os.makedirs(opt.www)

    return opt


def str2list(v):
    if len(v) == 2 or v is None:
        return []
    return v[1:-1].replace(' ', '').replace('\"', '').replace('\'', '').split(',')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
