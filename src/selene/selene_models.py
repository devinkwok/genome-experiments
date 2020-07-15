"""
Code for DeepSEA models and Selene framework from 
https://github.com/FunctionLab/selene/tree/master/tutorials
"""
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from ae.autoencoder import ReverseComplementConv1d, Autoencoder
from ae.load import load_model


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

    def __init__(self, sequence_length, n_targets, channel_sizes=[2, 3, 6], channel_size_factor=160):
        super(DeeperDeepSEA, self).__init__()
        self.conv_kernel_size = 9
        self.pool_kernel_size = 4
        self.sequence_length = sequence_length
        self.n_targets = n_targets
        self.n_units = len(channel_sizes)

        self.channel_sizes = [x * channel_size_factor for x in channel_sizes]
        self.channel_sizes.insert(0, 4)  # first dimension is 4

        self.conv_net = self._create_conv_net(self.channel_sizes)
        self.conv_output_channels = self.channel_sizes[-1]
        self.classifier = self._create_classifier()

    def _create_conv_net(self, channel_sizes):
        conv_net_dict = OrderedDict()
        for i in range(self.n_units):
            conv_net_dict['convA' + str(i)] = nn.Conv1d(channel_sizes[i], channel_sizes[i + 1], kernel_size=self.conv_kernel_size)
            conv_net_dict['reluA' + str(i)] = nn.ReLU(inplace=True)
            conv_net_dict['convB' + str(i)] = nn.Conv1d(channel_sizes[i + 1], channel_sizes[i + 1], kernel_size=self.conv_kernel_size)
            conv_net_dict['reluB' + str(i)] = nn.ReLU(inplace=True)
            conv_net_dict['maxpool' + str(i)] = nn.MaxPool1d(kernel_size=self.pool_kernel_size, stride=self.pool_kernel_size)
            conv_net_dict['batchnorm' + str(i)] = nn.BatchNorm1d(channel_sizes[i + 1])
            if i >= 1:
                conv_net_dict['dropout' + str(i)] = nn.Dropout(p=0.2)
        return nn.Sequential(conv_net_dict)


    def _create_classifier(self):
        reduce_by = 2 * (self.conv_kernel_size - 1)
        hidden_length = self.sequence_length
        for i in range(self.n_units):
            hidden_length = np.floor((hidden_length - reduce_by) / float(self.pool_kernel_size))
        self._n_channels = int(hidden_length)
        classifier = nn.Sequential(
            nn.Linear(self.conv_output_channels * self._n_channels, self.n_targets),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(self.n_targets),
            nn.Linear(self.n_targets, self.n_targets),
            nn.Sigmoid())
        return classifier


    def forward(self, x):
        """
        Forward propagation of a batch.
        """
        out = self.conv_net(x)
        reshape_out = out.view(out.size(0), self.conv_output_channels * self._n_channels)
        predict = self.classifier(reshape_out)
        return predict


class ReverseComplementDeepSEA(DeeperDeepSEA):

    def __init__(self, sequence_length, n_targets, reverse_complement_flags, channel_sizes=[2, 3, 6], channel_size_factor=160):
        self.reverse_complement_flags = reverse_complement_flags
        super(ReverseComplementDeepSEA, self).__init__(sequence_length, n_targets, channel_sizes=channel_sizes, channel_size_factor=channel_size_factor)


    def _create_conv_net(self, channel_sizes):
        conv_net_dict = OrderedDict()
        for i in range(self.n_units):
            conv_net_dict['convA' + str(i)] = ReverseComplementConv1d(channel_sizes[i], channel_sizes[i + 1], kernel_size=self.conv_kernel_size,
                    reverse=self.reverse_complement_flags[i*2][0], complement=self.reverse_complement_flags[i*2][1])
            conv_net_dict['reluA' + str(i)] = nn.ReLU(inplace=True)
            conv_net_dict['convB' + str(i)] = ReverseComplementConv1d(channel_sizes[i + 1], channel_sizes[i + 1], kernel_size=self.conv_kernel_size,
                    reverse=self.reverse_complement_flags[i*2 + 1][0], complement=self.reverse_complement_flags[i*2 + 1][1])
            conv_net_dict['reluB' + str(i)] = nn.ReLU(inplace=True)
            conv_net_dict['maxpool' + str(i)] = nn.MaxPool1d(kernel_size=self.pool_kernel_size, stride=self.pool_kernel_size)
            conv_net_dict['batchnorm' + str(i)] = nn.BatchNorm1d(channel_sizes[i + 1])
            if i >= 1:
                conv_net_dict['dropout' + str(i)] = nn.Dropout(p=0.2)
        return nn.Sequential(conv_net_dict)


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
