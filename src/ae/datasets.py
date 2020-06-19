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

# map style dataset holding entire sequence in memory
# 
class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, fasta_filename, subseq_len):
        bioseq = SeqIO.read(fasta_filename, "fasta")
        no_gaps = bioseq.seq.ungap('N').ungap('n')
        seq_array = [
            no_gaps,
            no_gaps[::-1],
            no_gaps.complement(),
            no_gaps.reverse_complement()
            ]
        self.seqs = [bioseq_to_np(x) for x in seq_array]
        self.subseq_len = subseq_len

    @property
    def subseq_len(self):
        return self.__subseq_len

    @subseq_len.setter
    def subseq_len(self, value):
        self.__subseq_len = value
        self._seq_len = len(self.seqs[0]) - self.__subseq_len + 1


    def __len__(self):
        return self._seq_len * 4


    def __getitem__(self, index):
        seq_type = int(index / self._seq_len)
        seq_pos = index % self._seq_len
        subseq = self.seqs[seq_type][seq_pos : seq_pos + self.subseq_len]
        return torch.LongTensor(subseq)


def bioseq_to_np(bioseq):
    str_array = np.array(bioseq)
    int_array = np.empty(len(bioseq), dtype='int8')
    for base, index in BASE_TO_INDEX.items():
        match = (str_array == base)
        int_array[match] = index
    return int_array
