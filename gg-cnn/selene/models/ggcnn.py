"""
An example model that has double the number of convolutional layers
that DeepSEA (Zhou & Troyanskaya, 2015) has. Otherwise, the architecture
is identical to DeepSEA.

We make no claims about the performance of this model. It is being stored
in `utils` so it can be easily loaded in the Jupyter notebook tutorials
for Selene, and may be removed in the future.

When making a model architecture file of your own, please review this
file in its entirety. In addition to the model class, Selene expects
that `criterion` and `get_optimizer(lr)` are also specified in this file.
"""
import numpy as np
import torch
import torch.nn as nn


def identity(x):
    return x

def complement(x):
    return torch.flip(x, [1])

def reverse(x):
    return torch.flip(x, [2])

def reverse_complement(x):
    return torch.flip(x, [1, 2])

def swap(x, dim=1):
    a, b = torch.split(x, x.shape[dim] // 2, dim=dim)
    return torch.cat([b, a], dim=dim)


class GroupConv1d(nn.Conv1d):

    def __init__(self, in_channels, out_channels, kernel_size, in_from_rci=False,
                stride=1, padding=0, dilation=1, groups=1, bias=True,
                do_reverse=True, do_complement=True):

        transforms = [identity]
        if do_reverse and do_complement:
            if in_from_rci:
                transforms += [lambda x: swap(reverse_complement(x))]
            else:
                transforms += [reverse_complement]
        if do_reverse:
            if in_from_rci:
                transforms += [lambda x: reverse(swap(x))]
            else:
                transforms += [reverse]
        if do_complement:
            transforms += [complement]
        assert out_channels % len(transforms) == 0

        super().__init__(in_channels, out_channels // len(transforms), kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self._transforms = transforms
        self.out_channels = out_channels

    # output channels are ordered as (normal, reverse_complement, reverse, complement)
    # complementation is still meaningful by flipping (reversing) the channel dimension
    def forward(self, x):
        return torch.cat([t(super(GroupConv1d, self).forward(t(x)))
                        for t in self._transforms], dim=1)


class DeepSEA(nn.Module):
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

    def __init__(self, sequence_length=1000, n_targets=919, hidden_size_scaling=1,
                data_augmentation='none', group_conv_type='none'):
        super(DeepSEA, self).__init__()
        conv_kernel_size = 8
        pool_kernel_size = 4
        self.augments = [identity]
        if data_augmentation == 'rc':
            self.augments.append(reverse_complement)
        elif data_augmentation == 'all':
            self.augments.append(reverse)
            self.augments.append(complement)
            self.augments.append(reverse_complement)
        do_reverse, do_complement = False, False
        if group_conv_type == 'all':
            do_reverse, do_complement = True, True

        self.conv_net = nn.Sequential(
            GroupConv1d(4, 320, kernel_size=conv_kernel_size, in_from_rci=False,
                do_reverse=do_reverse, do_complement=do_complement),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.BatchNorm1d(320),
            nn.Dropout(p=0.2),

            GroupConv1d(320, 480, kernel_size=conv_kernel_size, in_from_rci=True,
                do_reverse=do_reverse, do_complement=do_complement),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.BatchNorm1d(480),
            nn.Dropout(p=0.2),

            GroupConv1d(480, 960, kernel_size=conv_kernel_size, in_from_rci=True,
                do_reverse=do_reverse, do_complement=do_complement),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(960),
            nn.Dropout(p=0.2))

        reduce_by = 1 * (conv_kernel_size - 1)
        pool_kernel_size = float(pool_kernel_size)
        self._n_channels = int(
            np.floor(
                (np.floor(
                    (sequence_length - reduce_by) / pool_kernel_size)
                 - reduce_by) / pool_kernel_size)
            - reduce_by)
        self.classifier = nn.Sequential(
            nn.Linear(960 * self._n_channels, n_targets),
            nn.Sigmoid())

    def forward(self, x):
        """
        Forward propagation of a batch.
        """
        if self.training and len(self.augments) > 1:
            random_transform = torch.randint(len(self.augments), [1], dtype=torch.int).item()
            x = self.augments[random_transform](x)

        out = self.conv_net(x)
        reshape_out = out.view(out.size(0), 960 * self._n_channels)
        predict = self.classifier(reshape_out)
        return predict

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
