import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class Scheduler:
    def __init__(self, optimizer, args, train_steps):
        self.optimizer = optimizer
        self.scheduler_type = args.lradj

        self.step_size = args.step_size
        self.lr_decay = args.lr_decay
        self.min_lr = args.min_lr
        self.mode = args.mode
        self.train_epochs = args.train_epochs
        self.train_steps = train_steps
        self.pct_start = args.pct_start

        if self.scheduler_type is None:
            self.scheduler = None

        elif self.scheduler_type == 'reduce':
            _mode = 'min' if self.mode == 0 else 'max'
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=_mode, factor=self.lr_decay, patience=self.step_size, min_lr=self.min_lr)

        elif self.scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.step_size, eta_min=self.min_lr)

        elif self.scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.lr_decay)

        elif self.scheduler_type == 'type1':
            self.scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.5 ** (epoch // 1))

        elif self.scheduler_type == 'type2':
            lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
            lr_lambda = {epoch: lr / args.learning_rate for epoch, lr in lr_adjust.items()}
            self.scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_lambda.get(epoch, 1.0))

        elif self.scheduler_type == 'type3':
            self.scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 if epoch < 3 else 0.9 ** ((epoch - 3) // 1))

        elif self.scheduler_type == 'cosine2':
            self.scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 + math.cos(epoch / args.train_epochs * math.pi)) / 2)

        elif self.scheduler_type == 'TST':
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, steps_per_epoch=self.train_steps, epochs=self.train_epochs,
                max_lr=args.learning_rate, pct_start=self.pct_start
            )

        else:
            raise NotImplementedError

        if self.scheduler is not None:
            self.last_lr = self.scheduler._last_lr[0]
        else:
            self.last_lr = optimizer.param_groups[0]['lr']
        print(f'Initial learning rate: {self.last_lr}')

    def get_lr(self):
        return self.last_lr

    def step(self, val_loss=None, epoch=None, verbose=True):
        if self.scheduler_type is None or self.scheduler_type == 'none':
            return
        elif self.scheduler_type == 'reduce':
            self.scheduler.step(val_loss, epoch)
        elif epoch is not None:
            self.scheduler.step(epoch)
        else:
            self.scheduler.step()
        self.lr_info(verbose=verbose)

    def lr_info(self, verbose=True):
        last_lr = self.scheduler._last_lr[0]
        if last_lr != self.last_lr:
            if verbose:
                print(f'Updating learning rate from {self.last_lr} to {last_lr}')
            self.last_lr = last_lr


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, 'checkpoint.pth'))
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure(figsize=(4, 3), dpi=100)
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


class EvalAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super().__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        try:
            values = eval(values)
        except:
            try:
                values = eval(values.lower().capitalize())
            except:
                pass
        setattr(namespace, self.dest, values)


class PParameter(nn.Parameter):
    def __repr__(self):
        tensor_type = str(self.data.type()).split('.')[-1]
        size_str = " x ".join(map(str, self.shape))
        return f"Parameter containing: [{tensor_type} of size {size_str}]"


def ensure_path(path):
    os.makedirs(path, exist_ok=True)


def pv(msg, verbose):
    if verbose:
        print(msg)
