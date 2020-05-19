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
def slice_seq(seq1h_tensor, length, overlap=0):

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
