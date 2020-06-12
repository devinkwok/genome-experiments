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
        self.ae = Autoencoder(kernel_len=5, latent_len=2, seq_len=17,
                    seq_per_batch=7, input_dropout_freq=0.0, latent_noise_std=0.0)
        self.filename = "data/ref_genome/test.fasta"
        self.train_loader, self.valid_loader = load_data(self.ae, self.filename, split_prop=0.2)
        self.shape = (5, 4, 3)
        self.ones = torch.ones(*self.shape)
        self.zeros = torch.zeros(*self.shape)


    def test_load_data(self):
        for x in self.train_loader:
            self.assertEqual(x.shape, (self.ae.seq_per_batch, data_in.N_BASE, self.ae.seq_len))
            break  # only test the first batch
        for x in self.valid_loader:
            self.assertEqual(x.shape, (self.ae.seq_per_batch, data_in.N_BASE, self.ae.seq_len))
            break  # only test the first batch


    def test_forward(self):
        for x in self.train_loader:
            latent = self.ae.encode(x)
            reconstructed = self.ae.decode(latent)
            latent_shape = (self.ae.seq_per_batch, self.ae.latent_len, self.ae.seq_len - (self.ae.kernel_len - 1))
            self.assertEqual(latent.shape, latent_shape)
            self.assertEqual(reconstructed.shape, x.shape)
            break  # only test the first batch


    def test_predict(self):
        for x in self.train_loader:
            seq = predict(x)
            npt.assert_array_equal(seq, x)
            seq = predict(self.ae.forward(x)[0])
            npt.assert_array_equal(torch.sum(seq, dim=1), torch.ones(x.shape[0], x.shape[2]))
            break  # only test the first batch


    def test_evaluate(self):
        for x in self.train_loader:
            self.ae.eval()
            accuracy, error_indexes = evaluate(self.ae, x, predict(self.ae.forward(x)[0]))
            self.assertAlmostEqual(accuracy, 1.0)
            self.assertEqual(x[error_indexes[0],:, error_indexes[1]].nelement(), 0)

            accuracy, error_indexes = evaluate(self.ae, x, data_in.complement(
                predict(self.ae.forward(x)[0])))
            flattened = x.permute(0, 2, 1).reshape((x.shape[0]*x.shape[2], x.shape[1]))
            self.assertAlmostEqual(accuracy, 0)
            npt.assert_array_equal(flattened, x[error_indexes[0],:, error_indexes[1]])

            one_error = predict(self.ae.forward(x)[0])
            index_0, index_1 = 0, 1
            error = one_error[index_0, :, index_1].clone().detach()
            one_error[index_0, :, index_1] = error.flip(0)
            accuracy, error_indexes = evaluate(self.ae, x, one_error)
            self.assertLess(accuracy, 1.0)
            self.assertEqual(error_indexes[0], index_0)
            self.assertEqual(error_indexes[1], index_1)

            break  # only test the first batch


    def test_SeqDropout(self):
        dropout = SeqDropout(0.0)
        npt.assert_array_equal(dropout(self.ones), self.ones)

        dropout = SeqDropout(1.0)
        npt.assert_array_equal(dropout(self.ones), self.zeros)

        dropout = SeqDropout(0.5)
        self.assertLess(torch.sum(dropout(self.ones)), torch.sum(self.ones))
        self.assertGreater(torch.sum(dropout(self.ones)), 0)
        position_wise_sum = torch.sum(dropout(self.ones), dim=1)
        sums = torch.logical_or(position_wise_sum == 0., position_wise_sum == 4.)
        self.assertTrue(torch.all(sums))


    def test_GaussianNoise(self):
        # make size large enough to make std and mean unlikely to be off
        shape = (100, 4, 100)
        ones = torch.ones(*shape)

        noise = GaussianNoise(0)
        npt.assert_array_equal(noise(ones), ones)

        noise = GaussianNoise(0.5)
        std, mean = torch.std_mean(noise(ones))
        self.assertAlmostEqual(mean.item(), 1, places=1)
        self.assertAlmostEqual(std.item(), 0.5, places=1)
        noise.eval()
        npt.assert_array_equal(noise(ones), ones)


    def test_NeighbourDistanceLoss(self):
        x = torch.rand(self.shape)
        z = torch.rand(self.shape)

        nd_loss = NeighbourDistanceLoss(0.0)
        bce_loss = nn.BCELoss()
        self.assertEqual(nd_loss(x, z, self.ones).item(), bce_loss(x, z).item())

        nd_loss = NeighbourDistanceLoss(0.5)
        self.assertEqual(nd_loss(x, z, self.ones).item(), bce_loss(x, z).item() * 0.5)

        nd_loss = NeighbourDistanceLoss(1.0)
        self.assertEqual(nd_loss(x, z, self.ones).item(), 0.0)

        y = torch.arange(self.ones.nelement()).reshape(self.shape).float()
        self.assertEqual(nd_loss(x, z, y).item(), 1.0)


    def test_MultilayerEncoder(self):
        seq_per_batch = 10
        latent_len = 100
        ae = MultilayerEncoder(kernel_len=3, latent_len=latent_len, seq_len=128, seq_per_batch=seq_per_batch,
                    input_dropout_freq=0.05, latent_noise_std=0.2, hidden_len=10, pool_size=2,
                    n_conv_and_pool=2, n_conv_before_pool=2, n_linear=2)
        train_loader, _ = load_data(ae, self.filename, split_prop=0.05)
        for x in train_loader:
            reconstructed, latent = ae.forward(x)
            self.assertEqual(latent.shape, (seq_per_batch, latent_len))
            self.assertEqual(reconstructed.shape, x.shape)
            break


if __name__ == '__main__':
    unittest.main()
