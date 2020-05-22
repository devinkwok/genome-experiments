import sys
sys.path.append('./src/ae/')

import math
import unittest

import torch
import numpy as np
import numpy.testing as npt

from data_in import *


class Test_Data_In(unittest.TestCase):


    def setUp(self):
        self.test_seq = ['A', 'a', 'G', 'g', 'C', 'c', 'T', 't', 'N', 'n',]
        self.test_seq_len = len(self.test_seq)
        self.test_complement = ['T', 't', 'C', 'c', 'G', 'g', 'A', 'a', 'N', 'n',]
        self.test_np = np.array([
                [1, 0, 0, 0], [1, 0, 0, 0],
                [0, 1, 0, 0], [0, 1, 0, 0],
                [0, 0, 1, 0], [0, 0, 1, 0],
                [0, 0, 0, 1], [0, 0, 0, 1],
                [0, 0, 0, 0], [0, 0, 0, 0],
                ], dtype=np.float32)
        self.test_tensor = torch.tensor(self.test_np)
        self.filename = "data/ref_genome/test_short.fasta"
        self.filename = "data/ref_genome/test.fasta"
        self.seq_len = 17
        self.overlap = 4
        self.dataset = SeqData.from_file(self.filename, seq_len=self.seq_len, overlap=self.overlap, do_cull=False)


    def test_seq_to_1h(self):
        seq = seq_to_1h(self.test_seq)
        self.assertEqual(seq.shape, self.test_np.shape)
        npt.assert_array_almost_equal(seq, self.test_np)


    def test_reverse(self):
        seq = reverse(self.test_tensor)
        self.assertEqual(seq.shape, self.test_np.shape)
        npt.assert_array_almost_equal(seq, self.test_np[::-1,])


    def test_complement(self):
        seq = complement(self.test_tensor)
        self.assertEqual(seq.shape, self.test_np.shape)
        npt.assert_array_almost_equal(seq, seq_to_1h(self.test_complement))


    def test_slice_seq(self):
        slice_len = 3  # must be >= 2

        with self.assertRaises(ValueError) as context:
            seq = slice_seq(self.test_tensor, length=slice_len, overlap=2)
        self.assertTrue(" cannot be more than half of length: " in str(context.exception))

        seq = slice_seq(self.test_tensor, length=slice_len, overlap=0)
        
        n_slices = int(math.ceil(self.test_seq_len / slice_len))
        new_shape = (n_slices, slice_len, N_BASE)
        self.assertEqual(seq.shape, new_shape)
        padded = np.pad(self.test_np, [(0, n_slices * slice_len - self.test_seq_len), (0, 0)])
        npt.assert_array_almost_equal(seq, padded.reshape(new_shape))

        overlap = 1
        stride = slice_len - overlap
        seq = slice_seq(self.test_tensor, length=slice_len, overlap=overlap)
        n_slices = int(math.ceil(self.test_seq_len / stride))
        self.assertEqual(seq.shape, (n_slices, slice_len, N_BASE))
        for i in range(n_slices):
            start = i * stride
            end = start + slice_len
            npt.assert_array_almost_equal(seq[i, :, :], padded[start:end, :])


    def test_pad_seq(self):
        for before in range(0, 5):
            for after in range(0, 11):
                seq = pad_seq(self.test_tensor, self.test_seq_len + after, offset=before)
                self.assertEqual(seq.shape[0], self.test_seq_len + before + after)
                npt.assert_array_almost_equal(seq[before:self.test_seq_len + before, :], self.test_tensor)
                npt.assert_array_almost_equal(seq[0:before:, :], np.zeros((before, N_BASE)))
                npt.assert_array_almost_equal(seq[self.test_seq_len + before:, :], np.zeros((after, N_BASE)))


    def test_cull_empty(self):
        offset = int(self.test_seq_len / 2)
        seq = pad_seq(self.test_tensor, 2 * self.test_seq_len + offset , offset=offset)
        seq = slice_seq(seq, length=self.test_seq_len, overlap=0)
        culled_zeros = cull_empty(seq, base_freq=0).shape[0]
        self.assertLessEqual(culled_zeros, seq.shape[0])
        culled_all = cull_empty(seq, base_freq=1).shape[0]
        self.assertEqual(culled_all, 0)
        culled_some = cull_empty(seq, base_freq=0.49).shape[0]
        self.assertEqual(culled_some + 1, culled_zeros)


    def test_SeqData(self):
        # multiply by 4 to account for data augmentation
        test_len = len(read_seq(self.filename)) * 4
        n_seq = math.ceil(test_len / (self.seq_len - self.overlap))
        self.assertEqual(self.dataset.seq.shape, (n_seq, N_BASE, self.seq_len))


    def test_SeqData_split(self):
        split_1, split_2 = self.dataset.split(split_prop=0.5, shuffle=False)
        n = len(self.dataset)
        if n % 2 == 0:
            self.assertEqual(split_1.seq.shape, split_2.seq.shape)
        else:
            self.assertEqual(split_1.seq.shape, split_2.seq[:-1,:,:].shape)
        npt.assert_array_equal(split_1.seq[0:int(n/2)], self.dataset.seq[0:int(n/2)])

        split_1, split_2 = self.dataset.split(split_prop=0.0, shuffle=False)
        self.assertEqual(split_2.seq.shape, self.dataset.seq.shape)
        npt.assert_array_equal(split_2.seq, self.dataset.seq)
        split_1, split_2 = self.dataset.split(split_prop=0.7, shuffle=False)
        self.assertEqual(len(split_1) + len(split_2), n)

        split_1, split_2 = self.dataset.split(split_prop=1.0, shuffle=True)
        self.assertEqual(split_1.seq.shape, self.dataset.seq.shape)
        self.assertFalse((split_1.seq == self.dataset.seq).all())


if __name__ == '__main__':
    unittest.main()
