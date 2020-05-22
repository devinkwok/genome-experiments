import math

import torch
import numpy as np
from Bio import SeqIO


# 1h means one hot
# complement reverses order of 1h columns

BASE_TO_INDEX = {
    'A': 0, 'a': 0,
    'G': 1, 'g': 1,
    'C': 2, 'c': 2,
    'T': 3, 't': 3,
    }

N_BASE = 4


class SeqData(torch.utils.data.Dataset):


    def __init__(self, tensor):
        self.seq = tensor


    @classmethod
    def from_file(cls, filename, seq_len=1, overlap=0, do_cull=False, cull_threshold=0.05):
        bio_seq = read_seq(filename)
        seq1h_tensor = seq_to_1h(bio_seq)

        # augment data with reverse and complement
        augmented_seq = torch.cat((seq1h_tensor, reverse(seq1h_tensor)), 0)
        augmented_seq = torch.cat((augmented_seq, complement(augmented_seq)), 0)

        # use overlap of (window size - 1) to ensure every position has convolution applied once
        input_data = slice_seq(augmented_seq, length=seq_len, overlap=overlap)
        if do_cull:
            input_data = cull_empty(input_data, base_freq=cull_threshold)
        # TODO: this is a quick fix to get dimensions in correct order for Conv1D
        input_data = input_data.permute(0, 2, 1)
        return cls(input_data)


    @classmethod
    def from_SeqData(cls, obj_SeqData):
        return cls(obj_SeqData.seq)


    def __len__(self):
        return self.seq.shape[0]


    def __getitem__(self, index):
        return self.seq[index,:,:]


    def split(self, split_prop=0.5, shuffle=False):
        n = self.__len__()
        if shuffle:
            indexes = torch.randperm(n)
        else:
            indexes = torch.arange(n)
        split_1 = indexes[:int(split_prop * n)]
        split_2 = indexes[int(split_prop * n):]
        return SeqData(self.seq[split_1]), SeqData(self.seq[split_2])


def read_seq(filename):
    return SeqIO.read(filename, 'fasta')


def seq_to_1h(bio_seq):
    np_seq = np.array(bio_seq)
    n_indexes = len(set(BASE_TO_INDEX.values()))
    one_hot = np.zeros((len(np_seq), n_indexes), dtype=np.float32)
    for base, index in BASE_TO_INDEX.items():
        one_hot[:, index] = one_hot[:, index] + (np_seq == base)
    return torch.tensor(one_hot, dtype=torch.float32)


def reverse(seq1h_tensor):
    return seq1h_tensor.flip([0])


def complement(seq1h_tensor):
    return seq1h_tensor.flip([1])


# divide sequence into subsequences of length or shorter (pad remaining with empty)
# with overlap for convolution
def slice_seq(seq1h_tensor, length=1, overlap=0):

    # can't overlap more than 1/2 of sequence for ease of implementation
    if overlap > length / 2:
        raise ValueError("overlap of " + str(overlap) + " cannot be more than half of length: " + str(length))

    stride = length - overlap
    n_sub = math.ceil(seq1h_tensor.shape[0] / stride)

    # pad sequence with empty at end and reshape into subsequences
    # include extra empty entry subsequence for next step
    padded_tensor = pad_seq(seq1h_tensor, (n_sub + 1) * stride)
    padded_tensor = padded_tensor.view(n_sub + 1, stride, N_BASE)

    # duplicate the tensor, remove first and last subsequences,
    # then reshape to get subsequences [12, 23, 34, ... , n(empty)]
    sub_seq1h_tensor = torch.repeat_interleave(padded_tensor, 2, dim=0)
    sub_seq1h_tensor = sub_seq1h_tensor[1:-1,:,:]
    sub_seq1h_tensor = sub_seq1h_tensor.view(n_sub, stride * 2, N_BASE)

    # trim the excess overlap at the end of each subsequence
    return sub_seq1h_tensor[:, :length , :]


def pad_seq(seq1h_tensor, target_len, offset=0):
    tensor_len = seq1h_tensor.shape[0]
    assert target_len >= tensor_len
    padded_tensor = torch.zeros(target_len + offset, N_BASE)
    padded_tensor[offset:tensor_len + offset, :] = seq1h_tensor
    return padded_tensor


# get rid of subsequences which have less or equal to base_freq proportion non-empty
def cull_empty(sub_seq1h_tensor, base_freq=0.0):
    nonzero = torch.sum(sub_seq1h_tensor, dim=(1, 2))
    freqs = nonzero / sub_seq1h_tensor.shape[1]
    return sub_seq1h_tensor[freqs > base_freq]
