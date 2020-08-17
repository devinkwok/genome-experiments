import timeit
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product

from selene_sdk.sequences.sequence import sequence_to_encoding
from Bio.Seq import Seq


BASE_TO_INDEX = {
    'A': 0, 'a': 0,
    'G': 1, 'g': 1,
    'C': 2, 'c': 2,
    'T': 3, 't': 3,
    'N': 4, 'n': 4,
    }
EMPTY_INDEX = 4
INDEX_TO_BASE = ['A', 'G', 'C', 'T']
N_BASE = 4

if __name__ == '__main__':
    n_tries = 100
    do_print_memory_use = False
    dev = torch.device('cuda')

    if do_print_memory_use:
        n_tries = 1

    def print_memory_use():
        if do_print_memory_use:
            print(torch.cuda.memory_allocated(), torch.cuda.memory_cached())

    def test(fn_list, *args_lists):
        for args in product(*args_lists):
            for fn in fn_list:
                print(timeit.timeit(lambda: fn(*args), number=n_tries), fn, args)

    # testing fns
    def sub_fn(bioseq):
        str_array = np.array(bioseq)
        int_array = np.empty(len(bioseq), dtype='uint8')
        for base, index in BASE_TO_INDEX.items():
            match = (str_array == base)
            # print(base, index, match, str_array)
            int_array[match] = index
        return int_array

    def bioseq_to_index(bioseq):
        int_array = sub_fn(bioseq)
        return torch.tensor(int_array, dtype=torch.uint8)

    def one_hot(index_sequence, indexes=range(N_BASE), dim=1):
        with torch.no_grad():
            return torch.stack([(index_sequence == i).float() for i in indexes], dim=dim)


    seq_short = "AGTACACTGGT" * 10
    seq_med = "AGTACACTGGT" * 100
    seq_long = "AGTACACTGGT" * 1000
    # test(
    #     [
    #         Seq,
    #         lambda x: np.array(Seq(x)),  # this is very slow
    #         lambda x: sub_fn(Seq(x)),
    #         lambda x: bioseq_to_index(Seq(x)),
    #         lambda x: one_hot(bioseq_to_index(Seq(x)).reshape(1, -1)),
    #         lambda x: sequence_to_encoding(x, BASE_TO_INDEX, INDEX_TO_BASE),
    #     ],
    #     [seq_short, seq_med, seq_long])


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

    class RCIConv1d(nn.Conv1d):

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
            return torch.cat([t(super(RCIConv1d, self).forward(t(x)))
                            for t in self._transforms], dim=1)

    class GroupConv1d(nn.Conv1d):

        def __init__(self, in_channels, out_channels, kernel_size, in_from_group=False,
                    stride=1, padding=0, dilation=1, groups=1, bias=True,
                    do_reverse=True, do_complement=True):

            super(GroupConv1d, self).__init__(in_channels, out_channels, kernel_size,
                    stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
            self.kernel_size = (kernel_size, 1)
            self.w = nn.Parameter(self.weight.view(self.weight.size(0), -1).t())
            self.unfold = nn.Unfold((self.kernel_size[0], 1))

        # output channels are ordered as (normal, reverse_complement, reverse, complement)
        # complementation is still meaningful by flipping (reversing) the channel dimension
        def forward(self, x):
            unfolded_x = self.unfold(x.unsqueeze_(-1)).transpose(1, 2)
            y = unfolded_x.matmul(self.w)
            if not (self.bias is None):
                y = y + self.bias
            return y.transpose(1, 2)

    def convolve(x, layer):
        y = layer(x)
        print_memory_use()
        del y

    test(
        [convolve],
        [
            one_hot(bioseq_to_index(Seq(seq_med)).reshape(1, -1)).to(dev),
            one_hot(bioseq_to_index(Seq(seq_long)).reshape(1, -1)).to(dev),
        ],
        [
            # nn.Conv1d(4, 40, 4).to(dev),
            # RCIConv1d(4, 40, 4, do_reverse=False, do_complement=False).to(dev),
            GroupConv1d(4, 40, 4).to(dev),
            # RCIConv1d(4, 40, 4, do_reverse=True, do_complement=True).to(dev),
            # RCIConv1d(4, 160, 4, do_reverse=True, do_complement=True).to(dev),
            # nn.Conv1d(4, 400, 40).to(dev),
            # RCIConv1d(4, 400, 40, do_reverse=False, do_complement=False).to(dev),
            GroupConv1d(4, 400, 40).to(dev),
            # RCIConv1d(4, 400, 40, do_reverse=True, do_complement=True).to(dev),
            # RCIConv1d(4, 1600, 40, do_reverse=True, do_complement=True).to(dev),
        ],)