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
        # self.filename = "data/ref_genome/test_short.fasta"
        self.filename = "data/ref_genome/test.fasta"
        self.ae.load_data(self.filename, validation_split=0.2)
        test_np = np.array([
                [1, 0, 0, 0], [1, 0, 0, 0],
                [0, 1, 0, 0], [0, 1, 0, 0],
                [0, 0, 1, 0], [0, 0, 1, 0],
                [0, 0, 0, 1], [0, 0, 0, 1],
                [0, 0, 0, 0], [0, 0, 0, 0],
                ], dtype=np.float32)
        test_tensor = data_in.pad_seq(torch.tensor(test_np), self.ae.seq_len * self.ae.seq_per_batch)
        test_tensor = test_tensor.repeat_interleave(self.ae.window_len)
        test_tensor = test_tensor.reshape((self.ae.seq_per_batch * self.ae.window_len, self.ae.seq_len, 4))
        self.test_tensor = test_tensor.permute(0, 2, 1)


    def test_load_data(self):
        for batch in self.ae.train_loader:
            self.assertEqual(batch.shape, (self.ae.seq_per_batch, data_in.N_BASE, self.ae.seq_len))
            break  # only test the first batch
        for batch in self.ae.valid_loader:
            self.assertEqual(batch.shape, (self.ae.seq_per_batch, data_in.N_BASE, self.ae.seq_len))
            break  # only test the first batch


    def test_forward(self):
        reconstructed = self.ae.forward(self.test_tensor).detach().numpy()
        self.assertEqual(reconstructed.shape, self.test_tensor.shape)
        # TODO testing for forward pass

        for batch in self.ae.train_loader:
            latent = self.ae.encode(batch)
            reconstructed = self.ae.decode(latent)
            latent_shape = (self.ae.seq_per_batch, self.ae.latent_len, self.ae.decodable_len)
            self.assertEqual(latent.shape, latent_shape)
            self.assertEqual(reconstructed.shape, batch.shape)
            break  # only test the first batch

    
    def test_predict(self):
        #FIXME finish
        # seq = self.ae.predict(self.test_tensor, empty_cutoff_prob=0.0)
        # npt.assert_array_equal(torch.sum(seq, dim=1), torch.ones(self.test_tensor[0], self.test_tensor[2]))
        pass

if __name__ == '__main__':
    unittest.main()
