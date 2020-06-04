import sys
sys.path.append('./src/ae/')

import math
import unittest

import torch
import numpy as np
import numpy.testing as npt

import data_in
from autoencoder import *

class Test_Autoencoder(unittest.TestCase):


    def setUp(self):
        self.ae = Autoencoder(window_len=5, latent_len=2, seq_len=17, seq_per_batch=7)
        self.filename = "data/ref_genome/test.fasta"
        self.train_loader, self.valid_loader = load_data(self.ae, self.filename, split_prop=0.2)


    def test_load_data(self):
        for x, x_true in self.train_loader:
            self.assertEqual(x.shape, (self.ae.seq_per_batch, data_in.N_BASE, self.ae.seq_len))
            break  # only test the first batch
        for x, x_true in self.valid_loader:
            self.assertEqual(x.shape, (self.ae.seq_per_batch, data_in.N_BASE, self.ae.seq_len))
            break  # only test the first batch


    def test_forward(self):
        for x, x_true in self.train_loader:
            latent = self.ae.encode(x)
            reconstructed = self.ae.decode(latent)
            latent_shape = (self.ae.seq_per_batch, self.ae.latent_len, self.ae.seq_len - (self.ae.window_len - 1))
            self.assertEqual(latent.shape, latent_shape)
            self.assertEqual(reconstructed.shape, x.shape)
            break  # only test the first batch


    def test_predict(self):
        for x, x_true in self.train_loader:
            seq = predict(x)
            npt.assert_array_equal(seq, x)
            seq = predict(self.ae.forward(x))
            npt.assert_array_equal(torch.sum(seq, dim=1), torch.ones(x.shape[0], x.shape[2]))
            break  # only test the first batch


    def test_evaluate(self):
        for x, x_true in self.train_loader:
            self.ae.eval()
            accuracy, error_indexes = evaluate(self.ae, x, predict(self.ae.forward(x)))
            self.assertAlmostEqual(accuracy, 1.0)
            self.assertEqual(x[error_indexes[0],:, error_indexes[1]].nelement(), 0)

            accuracy, error_indexes = evaluate(self.ae, x, data_in.complement(
                predict(self.ae.forward(x))))
            flattened = x.permute(0, 2, 1).reshape((x.shape[0]*x.shape[2], x.shape[1]))
            self.assertAlmostEqual(accuracy, 0)
            npt.assert_array_equal(flattened, x[error_indexes[0],:, error_indexes[1]])

            one_error = predict(self.ae.forward(x))
            index_0, index_1 = 0, 1
            error = one_error[index_0, :, index_1].clone().detach()
            one_error[index_0, :, index_1] = error.flip(0)
            accuracy, error_indexes = evaluate(self.ae, x, one_error)
            self.assertLess(accuracy, 1.0)
            self.assertEqual(error_indexes[0], index_0)
            self.assertEqual(error_indexes[1], index_1)

            break  # only test the first batch


    def test_dropout(self):
        shape = (5, 4, 3)
        ones = torch.ones(*shape)
        zeros = torch.zeros(*shape)

        dropout = SeqDropout(0.0)
        npt.assert_array_equal(dropout(ones), ones)

        dropout = SeqDropout(1.0)
        npt.assert_array_equal(dropout(ones), zeros)

        dropout = SeqDropout(0.5)
        self.assertLess(torch.sum(dropout(ones)), torch.sum(ones))
        self.assertGreater(torch.sum(dropout(ones)), 0)
        position_wise_sum = torch.sum(dropout(ones), dim=1)
        sums = torch.logical_or(position_wise_sum == 0., position_wise_sum == 4.)
        self.assertTrue(torch.all(sums))


if __name__ == '__main__':
    unittest.main()
