"""
Code for DeepSEA models and Selene framework from 
https://github.com/FunctionLab/selene/tree/master/tutorials
"""

import numpy as np
import torch
import torch.nn as nn

from ae.autoencoder import ReverseComplementConv1d, Autoencoder
from ae.train import load_model


class DeeperDeepSEA(nn.Module):
    """
    A deeper DeepSEA model architecture.

    Parameters
    ----------
    sequence_length : int
        The length of the sequences on which the model trains and and makes
        predictions.
    n_targets : int
        The number of targets (classes) to predict.

    Attributes
    ----------
    conv_net : torch.nn.Sequential
        The convolutional neural network component of the model.
    classifier : torch.nn.Sequential
        The linear classifier and sigmoid transformation components of the
        model.

    """

    def __init__(self, sequence_length, n_targets):
        super(DeeperDeepSEA, self).__init__()
        self.conv_kernel_size = 9
        self.pool_kernel_size = 4

        self.conv_net = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=self.conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(320, 320, kernel_size=self.conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=self.pool_kernel_size, stride=self.pool_kernel_size),
            nn.BatchNorm1d(320),

            nn.Conv1d(320, 480, kernel_size=self.conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(480, 480, kernel_size=self.conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=self.pool_kernel_size, stride=self.pool_kernel_size),
            nn.BatchNorm1d(480),
            nn.Dropout(p=0.2),

            nn.Conv1d(480, 960, kernel_size=self.conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(960, 960, kernel_size=self.conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(960),
            nn.Dropout(p=0.2))

        reduce_by = 2 * (self.conv_kernel_size - 1)
        self._n_channels = int(
            np.floor(
                (np.floor(
                    (sequence_length - reduce_by) / float(self.pool_kernel_size))
                 - reduce_by) / float(self.pool_kernel_size))
            - reduce_by)
        self.classifier = nn.Sequential(
            nn.Linear(960 * self._n_channels, n_targets),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(n_targets),
            nn.Linear(n_targets, n_targets),
            nn.Sigmoid())

    def forward(self, x):
        """
        Forward propagation of a batch.
        """
        out = self.conv_net(x)
        reshape_out = out.view(out.size(0), 960 * self._n_channels)
        predict = self.classifier(reshape_out)
        return predict


class ReverseComplementDeepSEA(DeeperDeepSEA):

    def __init__(self, sequence_length, n_targets, reverse_complement_flags):
        super(ReverseComplementDeepSEA, self).__init__(sequence_length, n_targets)

        self.conv_net = nn.Sequential(
            ReverseComplementConv1d(4, 320, kernel_size=self.conv_kernel_size,
                    reverse=reverse_complement_flags[0][0], complement=reverse_complement_flags[0][1]),
            nn.ReLU(inplace=True),
            ReverseComplementConv1d(320, 320, kernel_size=self.conv_kernel_size,
                    reverse=reverse_complement_flags[1][0], complement=reverse_complement_flags[1][1]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=self.pool_kernel_size, stride=self.pool_kernel_size),
            nn.BatchNorm1d(320),

            ReverseComplementConv1d(320, 480, kernel_size=self.conv_kernel_size,
                    reverse=reverse_complement_flags[2][0], complement=reverse_complement_flags[2][1]),
            nn.ReLU(inplace=True),
            ReverseComplementConv1d(480, 480, kernel_size=self.conv_kernel_size,
                    reverse=reverse_complement_flags[3][0], complement=reverse_complement_flags[3][1]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=self.pool_kernel_size, stride=self.pool_kernel_size),
            nn.BatchNorm1d(480),
            nn.Dropout(p=0.2),

            ReverseComplementConv1d(480, 960, kernel_size=self.conv_kernel_size,
                    reverse=reverse_complement_flags[4][0], complement=reverse_complement_flags[4][1]),
            nn.ReLU(inplace=True),
            ReverseComplementConv1d(960, 960, kernel_size=self.conv_kernel_size,
                    reverse=reverse_complement_flags[5][0], complement=reverse_complement_flags[5][1]),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(960),
            nn.Dropout(p=0.2))

# TODO need to correctly specify encoded channels and linear channels
# need to test imports load
class CopyKernelDeepSEA(DeeperDeepSEA):

    def __init__(self, sequence_length, n_targets, encode_model_config, keep_n_layers):
        super(CopyKernelDeepSEA, self).__init__(sequence_length, n_targets)

        self.encode_model = load_model(encode_model_config)
        self.encode_model.decapitate(keep_n_layers=keep_n_layers)

        layers = nn.ModuleDict(
            nn.Conv1d(4, 320, kernel_size=self.conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(320, 320, kernel_size=self.conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=self.pool_kernel_size, stride=self.pool_kernel_size),
            nn.BatchNorm1d(320),

            nn.Conv1d(320, 480, kernel_size=self.conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(480, 480, kernel_size=self.conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=self.pool_kernel_size, stride=self.pool_kernel_size),
            nn.BatchNorm1d(480),
            nn.Dropout(p=0.2),

            nn.Conv1d(480, 960, kernel_size=self.conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(960, 960, kernel_size=self.conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(960),
            nn.Dropout(p=0.2))

        #TODO remove excess lower layers
        # replace in_channels with n_encoded_channels
        # add to self.conv_net
        # calculate n_linear_channels

        self.classifier = nn.Sequential(
            nn.Linear(960 * n_linear_channels, n_targets),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(n_targets),
            nn.Linear(n_targets, n_targets),
            nn.Sigmoid())


    def forward(self, x):
        out = self.encode_model.encode(x, override_convert_to_onehot=True)
        return super.forward(out)


def criterion():
    """
    Specify the appropriate loss function (criterion) for this
    model.

    Returns
    -------
    torch.nn._Loss
    """
    return nn.BCELoss()

def get_optimizer(lr):
    """
    Specify an optimizer and its parameters.

    Returns
    -------
    tuple(torch.optim.Optimizer, dict)
        The optimizer class and the dictionary of kwargs that should
        be passed in to the optimizer constructor.

    """
    return (torch.optim.SGD,
            {"lr": lr, "weight_decay": 1e-6, "momentum": 0.9})
