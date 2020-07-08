import sys
sys.path.append('./src/seq_util/')

import math
import unittest

import torch
import numpy as np
import numpy.testing as npt
from Bio.Seq import Seq

from datasets import *


class Test_Data_In(unittest.TestCase):

    def setUp(self):
        self.test_seq = ['A', 'a', 'G', 'g', 'C', 'c', 'T', 't']
        self.int_seq = [0, 0, 1, 1, 2, 2, 3, 3]
        self.bioseq = Seq(''.join(self.test_seq))
        self.filename = "data/ref_genome/test_short.fasta"


    def test_bioseq_to_tensor(self):
        npt.assert_array_equal(bioseq_to_tensor(self.bioseq).numpy(), self.int_seq)


    def test_SequenceDataset(self):
        dataset = SequenceDataset(self.filename, 1)
        bioseq = SeqIO.read(self.filename, "fasta").seq.ungap('N').ungap('n')
        self.assertEqual(len(dataset), len(bioseq))
        SUBSEQ_LEN = 2
        dataset.seq_len = SUBSEQ_LEN
        self.assertEqual(len(dataset), (len(bioseq) - SUBSEQ_LEN + 1))
        dataset.seq_len = len(bioseq)


    def test_RandomRepeatSequence(self):
        n_batch = 3
        repeat_len = 2
        n_repeats = 4
        dataset = RandomRepeatSequence(10, n_batch, n_repeats, repeat_len)
        self.assertEqual(len(dataset), n_batch)
        seq = dataset[0]
        repeat_seq = seq[0:repeat_len]
        npt.assert_array_equal(repeat_seq, seq[repeat_len:repeat_len * 2])
