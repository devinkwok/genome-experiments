import sys
sys.path.append('./src/ae/')

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
        self.assertEqual(len(dataset), len(bioseq) * 4)
        SUBSEQ_LEN = 2
        dataset.subseq_len = SUBSEQ_LEN
        self.assertEqual(len(dataset), (len(bioseq) - SUBSEQ_LEN + 1) * 4)
        dataset.subseq_len = len(bioseq)
        npt.assert_array_equal(dataset.__getitem__(0), bioseq_to_tensor(bioseq))
        npt.assert_array_equal(dataset.__getitem__(2), bioseq_to_tensor(bioseq.complement()))
        npt.assert_array_equal(dataset.__getitem__(3), bioseq_to_tensor(bioseq.reverse_complement()))
