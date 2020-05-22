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
        self.ae.load_data(self.filename, validation_split=0.2)


    def test_load_data(self):
        for x, x_true in self.ae.train_loader:
            self.assertEqual(x.shape, (self.ae.seq_per_batch, data_in.N_BASE, self.ae.seq_len))
            break  # only test the first batch
        for x, x_true in self.ae.valid_loader:
            self.assertEqual(x.shape, (self.ae.seq_per_batch, data_in.N_BASE, self.ae.seq_len))
            break  # only test the first batch


    def test_forward(self):
        for x, x_true in self.ae.train_loader:
            latent = self.ae.encode(x)
            reconstructed = self.ae.decode(latent)
            latent_shape = (self.ae.seq_per_batch, self.ae.latent_len, self.ae.decodable_len)
            self.assertEqual(latent.shape, latent_shape)
            self.assertEqual(reconstructed.shape, x.shape)
            break  # only test the first batch


    def test_predict(self):
        for x, x_true in self.ae.train_loader:
            self.ae.empty_cutoff_prob = 0.0
            seq = self.ae.predict(x)
            npt.assert_array_equal(seq, x)
            seq = self.ae.predict(self.ae.forward(x))
            npt.assert_array_equal(torch.sum(seq, dim=1), torch.ones(x.shape[0], x.shape[2]))
            self.ae.empty_cutoff_prob = 1.0
            seq = self.ae.predict(self.ae.forward(x))
            npt.assert_array_equal(torch.sum(seq, dim=1), torch.zeros(x.shape[0], x.shape[2]))
            break  # only test the first batch


    def test_evaluate(self):
        for x, x_true in self.ae.train_loader:
            accuracy, error_indexes = self.ae.evaluate(x, self.ae.predict(self.ae.forward(x)))
            self.assertAlmostEqual(accuracy, 1.0)
            self.assertEqual(x[error_indexes[0],:, error_indexes[1]].nelement(), 0)

            self.ae.empty_cutoff_prob = 0.0
            accuracy, error_indexes = self.ae.evaluate(x, data_in.complement(
                self.ae.predict(self.ae.forward(x))))
            flattened = x.permute(0, 2, 1).reshape((x.shape[0]*x.shape[2], x.shape[1]))
            self.assertAlmostEqual(accuracy, 0)
            npt.assert_array_equal(flattened, x[error_indexes[0],:, error_indexes[1]])

            one_error = self.ae.predict(self.ae.forward(x))
            index_0, index_1 = 0, 1
            if (one_error[index_0, 0, index_1] == 1):
                one_error[index_0, 1, index_1] = 1
            else:
                one_error[index_0, 0, index_1] = 1
            accuracy, error_indexes = self.ae.evaluate(x, one_error)
            self.assertLess(accuracy, 1.0)
            self.assertEqual(error_indexes[0], index_0)
            self.assertEqual(error_indexes[1], index_1)

            break  # only test the first batch


if __name__ == '__main__':
    unittest.main()
