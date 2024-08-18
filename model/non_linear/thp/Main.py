import argparse
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim

import model.non_linear.thp.transformer.Constants as Constants
from model.non_linear.thp import Utils

from model.non_linear.thp.preprocess.Dataset import get_dataloader
from model.non_linear.thp.transformer.Models import Transformer
from tqdm import tqdm


def prepare_dataloader(opt):
    """ Load real_data and prepare dataloader. """

    def load_data(name, dict_name):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            num_types = data['dim_process']
            data = data[dict_name]
            return data, int(num_types)

    print('[Info] Loading train real_data...')
    train_data, num_types = load_data(opt.data + 'train.pkl', 'train')
    print('[Info] Loading dev real_data...')
    dev_data, _ = load_data(opt.data + 'dev.pkl', 'dev')
    print('[Info] Loading test real_data...')
    test_data, _ = load_data(opt.data + 'test.pkl', 'test')

    trainloader = get_dataloader(train_data, opt.batch_size, shuffle=True)
    testloader = get_dataloader(test_data, opt.batch_size, shuffle=False)
    return trainloader, testloader, num_types


def train_epoch(model, training_data, optimizer, pred_loss_func, opt):
    """ Epoch operation in training phase. """

    model.train()

    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    for batch in tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        """ prepare real_data """
        event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)

        """ forward """
        optimizer.zero_grad()

        enc_out, prediction = model(event_type, event_time)

        """ backward """
        # negative log-likelihood
        event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type)
        event_loss = -torch.sum(event_ll - non_event_ll)

        # type prediction
        pred_loss, pred_num_event = Utils.type_loss(prediction[0], event_type, pred_loss_func)

        # time prediction
        se = Utils.time_loss(prediction[1], event_time)

        # SE is usually large, scale it to stabilize training
        scale_time_loss = 100
        loss = event_loss + pred_loss + se / scale_time_loss
        loss.backward()

        """ update parameters """
        optimizer.step()

        """ note keeping """
        total_event_ll += -event_loss.item()
        total_time_se += se.item()
        total_event_rate += pred_num_event.item()
        total_num_event += event_type.ne(Constants.PAD).sum().item()
        # we do not predict_rmtpp the first event
        total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

    rmse = np.sqrt(total_time_se / total_num_pred)
    return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse


# train with exogenous messages
def train_with_exo_mae(model, training_data, optimizer, scheduler, pred_loss_func, opt):
    """ Start training. """
    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()
        likelihood, mark_error, time_error = train_epoch_with_exo_mae(model, training_data, optimizer, pred_loss_func,
                                                                      opt)
        print('  - (Training)    loglikelihood: {likelihood: 8.5f}, '
              'mark error: {mark_error: 8.5f}, time error: {time_error: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(likelihood=likelihood, mark_error=mark_error, time_error=time_error,
                      elapse=(time.time() - start) / 60))
        scheduler.step()


# endo_mask_out in training_data
def train_epoch_with_exo_mae(model, training_data, optimizer, pred_loss_func, opt):
    model.train()

    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_time_ae = 0  # absolute error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    for batch in tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        """ prepare real_data """
        event_time, time_gap, event_type, endo_mask = map(lambda x: x.to(opt.device), batch)

        """ forward """
        optimizer.zero_grad()

        enc_out, prediction = model(event_type, event_time)

        """ backward """
        # negative log-likelihood
        event_ll, non_event_ll = Utils.log_likelihood_exo(model, enc_out, event_time, event_type, endo_mask)
        event_loss = -torch.sum(event_ll - non_event_ll)

        # type prediction
        pred_loss, pred_num_event = Utils.type_loss_exo(prediction[0], event_type, pred_loss_func, endo_mask)

        # time prediction
        se = Utils.time_loss_exo(prediction[1], event_time, endo_mask)

        # SE is usually large, scale it to stabilize training
        scale_time_loss = 100
        loss = event_loss + pred_loss + se / scale_time_loss
        loss.backward()

        """ update parameters """
        optimizer.step()

        """ note keeping """
        total_event_ll += -event_loss.item()
        """ se           """
        total_time_se += se.item()
        """ ae           """
        ae = Utils.time_loss_ae(prediction[1], event_time)
        total_time_ae += ae.item()
        total_event_rate += pred_num_event.item()
        total_num_event += event_type.ne(Constants.PAD).sum().item()
        # we do not predict_rmtpp the first event
        total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

    mae = total_time_ae / total_num_pred
    # likelihood, mark_error, time_error
    return total_event_ll / total_num_event, 1 - (total_event_rate / total_num_pred), mae


def train(model, training_data, validation_data, optimizer, scheduler, pred_loss_func, opt):
    """ Start training. """
    valid_event_losses = []  # validation log-likelihood
    valid_pred_losses = []  # validation event type prediction accuracy
    valid_rmse = []  # validation event time prediction RMSE
    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()
        train_event, train_type, train_time = train_epoch(model, training_data, optimizer, pred_loss_func, opt)
        print('  - (Training)    loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=train_event, type=train_type, rmse=train_time, elapse=(time.time() - start) / 60))

        start = time.time()
        valid_event, valid_type, valid_time = eval_epoch(model, validation_data, pred_loss_func, opt)
        print('  - (Testing)     loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=valid_event, type=valid_type, rmse=valid_time, elapse=(time.time() - start) / 60))

        valid_event_losses += [valid_event]
        valid_pred_losses += [valid_type]
        valid_rmse += [valid_time]
        print('  - [Info] Maximum ll: {event: 8.5f}, '
              'Maximum accuracy: {pred: 8.5f}, Minimum RMSE: {rmse: 8.5f}'
              .format(event=max(valid_event_losses), pred=max(valid_pred_losses), rmse=min(valid_rmse)))

        # logging
        with open(opt.log, 'a') as f:
            f.write('{epoch}, {ll: 8.5f}, {acc: 8.5f}, {rmse: 8.5f}\n'
                    .format(epoch=epoch, ll=valid_event, acc=valid_type, rmse=valid_time))

        scheduler.step()


def eval_epoch_mae(model, validation_data, pred_loss_func, opt):
    """ Epoch operation in evaluation phase. """

    pred_event, true_event, pred_time, true_time = [], [], [], []

    model.eval()

    total_event_ll = 0  # cumulative event log-likelihood
    total_time_ae = 0  # absolute error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            if batch[0].size(0) < opt.batch_size:
                continue
            """ prepare real_data """
            event_time, time_gap, event_type, _ = map(lambda x: x.to(opt.device), batch)

            """ forward """
            enc_out, prediction = model(event_type, event_time)

            """ compute loss """
            event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type)
            event_loss = -torch.sum(event_ll - non_event_ll)
            _, pred_num = Utils.type_loss(prediction[0], event_type, pred_loss_func)
            ae = Utils.time_loss_ae(prediction[1], event_time)

            """ note keeping """
            total_event_ll += -event_loss.item()
            total_time_ae += ae.item()
            total_event_rate += pred_num.item()
            total_num_event += event_type.ne(Constants.PAD).sum().item()
            total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

            pred_event += [prediction[0][:, :-1, :]]
            true_event += [event_type[:, 1:] - 1]
            pred_time += [prediction[1][:, :-1]]
            true_time += [event_time[:, 1:] - event_time[:, :-1]]

    mae = total_time_ae / total_num_pred

    return pred_event, true_event, pred_time, true_time, \
           total_event_ll / total_num_event, 1 - (total_event_rate / total_num_pred), mae
