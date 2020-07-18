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


def get_conv_width(input_length, kernel_size, n_pools, pool_size, n_convs_per_pool=2, no_padding=False):
    reduce_by = 0
    if no_padding:
        reduce_by = n_convs_per_pool * (kernel_size - 1)
    hidden_length = input_length
    for i in range(n_pools):
        hidden_length = np.floor((hidden_length - reduce_by) / float(pool_size))
    return int(hidden_length)


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

    def __init__(self, sequence_length, n_targets, channel_sizes=[2, 3, 6],
                channel_size_factor=160, input_channels=4):
        super(DeeperDeepSEA, self).__init__()
        self.conv_kernel_size = 9
        self.pool_kernel_size = 4
        self.sequence_length = sequence_length
        self.n_targets = n_targets
        self.n_units = len(channel_sizes)

        self.channel_sizes = [x * channel_size_factor for x in channel_sizes]
        self.channel_sizes.insert(0, input_channels)  # first dimension is 4

        self.conv_output_channels = self.channel_sizes[-1]
        
        self._n_channels = get_conv_width(self.sequence_length, self.conv_kernel_size,
                                        self.n_units, self.pool_kernel_size, no_padding=True)

        self.conv_net = self._create_conv_net(self.channel_sizes)
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

    def __init__(self, sequence_length, n_targets, reverse_complement_prop, channel_sizes=[2, 3, 6], channel_size_factor=160):
        self.reverse_complement_prop = reverse_complement_prop
        super(ReverseComplementDeepSEA, self).__init__(sequence_length, n_targets, channel_sizes=channel_sizes, channel_size_factor=channel_size_factor)


    def _create_conv_net(self, channel_sizes):
        conv_net_dict = OrderedDict()
        for i in range(self.n_units):
            out_sizes = [int(round(p * channel_sizes[i + 1])) for p in self.reverse_complement_prop[i]]
            out_sizes.insert(0, channel_sizes[i + 1] - sum(out_sizes))  # remaining channels assumed to be normal
            out_sizes += [0] * (4 - len(out_sizes))
            for j in range(4):
                if (out_sizes[j] < 0):  # need to remove some channels
                    out_sizes[j + 1] += out_sizes[j]

            conv_net_dict['convA' + str(i)] = ReverseComplementConv1d(
                                channel_sizes[i], *out_sizes, kernel_size=self.conv_kernel_size)
            conv_net_dict['reluA' + str(i)] = nn.ReLU(inplace=True)
            conv_net_dict['convB' + str(i)] = ReverseComplementConv1d(
                                channel_sizes[i + 1], *out_sizes, kernel_size=self.conv_kernel_size)
            conv_net_dict['reluB' + str(i)] = nn.ReLU(inplace=True)
            conv_net_dict['maxpool' + str(i)] = nn.MaxPool1d(kernel_size=self.pool_kernel_size, stride=self.pool_kernel_size)
            conv_net_dict['batchnorm' + str(i)] = nn.BatchNorm1d(channel_sizes[i + 1])
            if i >= 1:
                conv_net_dict['dropout' + str(i)] = nn.Dropout(p=0.2)
        return nn.Sequential(conv_net_dict)


class CopyKernelDeepSEA(DeeperDeepSEA):


    # go from 0 up to encoder_n_layers in the encoder, then from deepsea_n_units to max in deepsea
    def __init__(self, sequence_length, n_targets, encoder_model_config,
                encoder_n_layers, deepsea_n_layers, channel_size_factor=160, retrain_encoder=False):
        encoder_model = load_model(encoder_model_config)
        encoder_model.decapitate(keep_n_layers=encoder_n_layers)
        encoder_n_pool, encoder_channels = 0, 0
        for name, layer in encoder_model.encode_layers.items():
            if 'pool' in name:
                encoder_n_pool += 1
            if 'conv' in name:
                encoder_channels = layer.out_channels

        encoder_length = get_conv_width(sequence_length, encoder_model_config['kernel_len'],
                encoder_n_pool, encoder_model_config['pool_size'], no_padding=False)
        deepsea_channels = [x * channel_size_factor for x in [2, 3, 6]]
        deepsea_channels = deepsea_channels[deepsea_n_layers:]


        super(CopyKernelDeepSEA, self).__init__(encoder_length, n_targets,
                channel_sizes=deepsea_channels, channel_size_factor=1, input_channels=encoder_channels)
        self.encoder_model = encoder_model
        self.retrain_encoder = retrain_encoder


    def forward(self, x):
        if self.retrain_encoder:
            out = self.encoder_model.encode(x, override_convert_to_onehot=True)
        else:
            with torch.no_grad():
                out = self.encoder_model.encode(x, override_convert_to_onehot=True)
        return super(CopyKernelDeepSEA, self).forward(out)


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
