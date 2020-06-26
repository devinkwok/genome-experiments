import math

import torch
import torch.nn.functional as F
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

    def __init__(self, fasta_filename, seq_len, stride=1, overlap=None, get_label=False, make_onehot=False):
        print("reading sequence from file...")
        bioseq = SeqIO.read(fasta_filename, "fasta")
        self.seq = bioseq.seq.ungap('N').ungap('n')
        self.augment_state = 0
        if overlap is None:
            self.stride = stride
        else:
            self.stride = seq_len - overlap
        self.seq_len = seq_len
        self.get_label = get_label
        self.make_onehot = make_onehot


    @property
    def seq_len(self):
        return self._seq_len


    @seq_len.setter
    def seq_len(self, value):
        self._seq_len = value
        self._total_len = int((len(self.seq) - self._seq_len + 1) / self.stride)


    def __len__(self):
        return self._total_len


    def __getitem__(self, index):
        subseq = self.seq[index * self.stride : index * self.stride + self.seq_len]
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
        x = bioseq_to_tensor(subseq)
        if self.make_onehot:
            with torch.no_grad():
                x = F.one_hot(x, num_classes=N_BASE).permute(1, 0).type(torch.float32)
        if self.get_label:
            return x, x
        else:
            return x


def bioseq_to_tensor(bioseq):
    str_array = np.array(bioseq)
    int_array = np.empty(len(bioseq), dtype='int8')
    for base, index in BASE_TO_INDEX.items():
        match = (str_array == base)
        int_array[match] = index
    return torch.LongTensor(int_array)
