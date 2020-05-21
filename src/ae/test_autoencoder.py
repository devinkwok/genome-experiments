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
        self.data_loader = self.ae.load_data(self.filename, cull_empty=False)


    def test_load_data_from_fasta(self):
        data_loader = self.ae.load_data(self.filename, cull_empty=False)

        # multiply by 4 to account for data augmentation
        test_len = len(data_in.read_seq(self.filename)) * 4
        n_seq = math.ceil(test_len / (self.ae.decodable_len))

        self.assertEqual(data_loader.dataset.shape, (n_seq, data_in.N_BASE, self.ae.seq_len))
        for batch in data_loader:
            self.assertEqual(batch.shape, (self.ae.seq_per_batch, data_in.N_BASE, self.ae.seq_len))
            break  # only test the first batch


    def test_forward(self):
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
        test_tensor = test_tensor.permute(0, 2, 1)
        latent, reconstructed = self.ae.forward(test_tensor)
        latent = latent.detach().numpy()
        reconstructed = reconstructed.detach().numpy()
        # TODO testing for forward pass

        for batch in self.data_loader:
            latent, reconstructed = self.ae.forward(batch)
            latent_shape = (self.ae.seq_per_batch, self.ae.latent_len, self.ae.decodable_len)
            self.assertEqual(latent.shape, latent_shape)
            self.assertEqual(reconstructed.shape, batch.shape)
            break  # only test the first batch


if __name__ == '__main__':
    unittest.main()