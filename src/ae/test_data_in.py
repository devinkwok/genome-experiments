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
        self.seq_len = len(self.test_seq)
        self.test_complement = ['T', 't', 'C', 'c', 'G', 'g', 'A', 'a', 'N', 'n',]
        self.test_np = np.array([
                [1, 0, 0, 0], [1, 0, 0, 0],
                [0, 1, 0, 0], [0, 1, 0, 0],
                [0, 0, 1, 0], [0, 0, 1, 0],
                [0, 0, 0, 1], [0, 0, 0, 1],
                [0, 0, 0, 0], [0, 0, 0, 0],
                ], dtype=np.float32)
        self.test_tensor = torch.tensor(self.test_np)


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
        
        n_slices = int(math.ceil(self.seq_len / slice_len))
        new_shape = (n_slices, slice_len, N_BASE)
        self.assertEqual(seq.shape, new_shape)
        padded = np.pad(self.test_np, [(0, n_slices * slice_len - self.seq_len), (0, 0)])
        npt.assert_array_almost_equal(seq, padded.reshape(new_shape))

        overlap = 1
        stride = slice_len - overlap
        seq = slice_seq(self.test_tensor, length=slice_len, overlap=overlap)
        n_slices = int(math.ceil(self.seq_len / stride))
        self.assertEqual(seq.shape, (n_slices, slice_len, N_BASE))
        for i in range(n_slices):
            start = i * stride
            end = start + slice_len
            npt.assert_array_almost_equal(seq[i, :, :], padded[start:end, :])


    def test_pad_seq(self):
        for before in range(0, 5):
            for after in range(0, 11):
                seq = pad_seq(self.test_tensor, self.seq_len + after, offset=before)
                self.assertEqual(seq.shape[0], self.seq_len + before + after)
                npt.assert_array_almost_equal(seq[before:self.seq_len + before, :], self.test_tensor)
                npt.assert_array_almost_equal(seq[0:before:, :], np.zeros((before, N_BASE)))
                npt.assert_array_almost_equal(seq[self.seq_len + before:, :], np.zeros((after, N_BASE)))


    def test_cull_empty(self):
        offset = int(self.seq_len / 2)
        seq = pad_seq(self.test_tensor, 2 * self.seq_len + offset , offset=offset)
        seq = slice_seq(seq, length=self.seq_len, overlap=0)
        culled_zeros = cull_empty(seq, base_freq=0).shape[0]
        self.assertLessEqual(culled_zeros, seq.shape[0])
        culled_all = cull_empty(seq, base_freq=1).shape[0]
        self.assertEqual(culled_all, 0)
        culled_some = cull_empty(seq, base_freq=0.49).shape[0]
        self.assertEqual(culled_some + 1, culled_zeros)


if __name__ == '__main__':
    unittest.main()
