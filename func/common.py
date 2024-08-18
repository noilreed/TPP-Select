import argparse
import torch
import datetime
import numpy as np

# exo_ratios=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
exo_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
# exo_ratios=[0.1, 0.5, 0.9]
train_test_ratio = [0.7, 0.3]


def get_ranking(order):
    temp = np.argsort(order)
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(order))
    return ranks


def thp_params():
    opt = argparse.Namespace
    opt.data = 'data/'
    opt.epoch = 60
    opt.batch_size = 16
    opt.d_model = 64
    opt.d_rnn = 256
    opt.d_inner_hid = 128
    opt.d_k = 16
    opt.d_v = 16
    opt.n_head = 4
    opt.n_layers = 4
    opt.dropout = 0.3
    opt.lr = 1e-4
    opt.smooth = 0.1
    opt.device = torch.device('cpu')
    return opt


DEFAULT_BATCH_SIZE = 32
DEFAULT_HIDDEN_SIZE = 16
DEFAULT_LEARN_RATE = 5e-5


def sahp_params():
    args = argparse.Namespace
    args.e = 20
    # args.e = 1
    args.b = DEFAULT_BATCH_SIZE
    args.lr = DEFAULT_LEARN_RATE
    args.hidden = DEFAULT_HIDDEN_SIZE
    args.d_model = DEFAULT_HIDDEN_SIZE
    args.atten_heads = 8
    args.pe = 'add'
    args.nLayers = 4
    args.dropout = 0.1
    args.cuda = 0
    args.train_ratio = 0.8
    args.lambda_l2 = 3e-4
    args.dev_ratio = 0.1
    args.early_stop_threshold = 1e-2
    args.log_dir = 'logs'
    args.save_model = False
    args.bias = False
    args.samples = 10
    args.model = 'sahp'
    args.task = 'retweet'
    return args


opt = thp_params()
args = sahp_params()


def nhp_params():
    settings = {
        'hidden_size': 32,
        'type_size': 2,
        'train_path': 'data/train.pkl',
        'dev_path': 'data/dev.pkl',
        'batch_size': 32,
        # 'epoch_num': 20,
        # 'epoch_num': 10,
        'epoch_num': 1,
        'current_date': datetime.date.today()
    }
    return settings


nhp_settings = nhp_params()
