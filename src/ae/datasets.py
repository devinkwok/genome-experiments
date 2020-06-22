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
        print("reading sequence from file...")
        bioseq = SeqIO.read(fasta_filename, "fasta")
        self.seq = bioseq.seq.ungap('N').ungap('n')
        self.subseq_len = subseq_len
        self.augment_state = 0


    @property
    def subseq_len(self):
        return self._subseq_len


    @subseq_len.setter
    def subseq_len(self, value):
        self._subseq_len = value
        self._seq_len = len(self.seq) - self._subseq_len + 1


    def __len__(self):
        return self._seq_len


    def __getitem__(self, index):
        subseq = self.seq[index:index + self.subseq_len]
        # randomly assign reverse, complement, or reverse complement
        self.augment_state += 1
        if self.augment_state == 1:
            subseq = subseq[::-1]
        elif self.augment_state == 2:
            subseq = subseq.complement()
        elif self.augment_state == 3:
            subseq = subseq.reverse_complement()
        else:
            self.augment_state = 0
        return bioseq_to_tensor(subseq)


def bioseq_to_tensor(bioseq):
    str_array = np.array(bioseq)
    int_array = np.empty(len(bioseq), dtype='int8')
    for base, index in BASE_TO_INDEX.items():
        match = (str_array == base)
        int_array[match] = index
    return torch.LongTensor(int_array)
