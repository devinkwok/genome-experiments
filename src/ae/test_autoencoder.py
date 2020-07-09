import sys
sys.path.append('./src/ae/')
sys.path.append('./src/seq_util/')

import math
import unittest

import torch
import numpy as np
import numpy.testing as npt

from train import load_data
from autoencoder import *
from datasets import SequenceDataset

class Test_Autoencoder(unittest.TestCase):


    def setUp(self):
        self.kernel_len = 5
        self.latent_len = 2
        self.seq_len = 20
        self.seq_per_batch = 7
        self.input_dropout_freq = 0.0
        self.latent_noise_std = 0.0
        self.loss_fn = NeighbourDistanceLoss(0.0)
        self.autoencoders = [
            Autoencoder(self.kernel_len, self.latent_len, self.seq_len, self.seq_per_batch,
                self.input_dropout_freq, self.latent_noise_std, self.loss_fn),
            MultilayerEncoder(self.kernel_len, self.latent_len, self.seq_len, self.seq_per_batch,
                    self.input_dropout_freq, self.latent_noise_std, self.loss_fn,
                    hidden_len=10, pool_size=2, n_conv_and_pool=2, n_conv_before_pool=2,
                    n_linear=2, hidden_dropout_freq=0.0),
        ]
        self.filename = "data/ref_genome/test.fasta"
        self.dataset = SequenceDataset(self.filename, seq_len=self.seq_len)
        self.train_loader, self.valid_loader = load_data(
                    self.autoencoders[0], self.dataset, split_prop=0.2, n_dataloader_workers=2)
        self.shape = (5, 4, 3)
        self.ones = torch.ones(*self.shape)
        self.zeros = torch.zeros(*self.shape)


    def test_load_data(self):
        for x in self.train_loader:
            self.assertEqual(x.shape, (self.seq_per_batch, self.seq_len))
            break  # only test the first batch
        for x in self.valid_loader:
            self.assertEqual(x.shape, (self.seq_per_batch * 2, self.seq_len))
            break  # only test the first batch


    def test_forward(self):
        for model in self.autoencoders:
            for x in self.train_loader:
                reconstructed, latent = model.forward(x)
                latent_shape = (model.seq_per_batch, model.latent_len, model.seq_len - (model.kernel_len - 1))
                #TODO need Multilayer encoder to be fully convolutional during training
                # self.assertEqual(latent.shape, latent_shape)
                self.assertEqual(reconstructed.shape, (x.shape[0], N_BASE, x.shape[1]))
                break  # only test the first batch


    def test_loss(self):
        for model in self.autoencoders:
            for x in self.train_loader:
                loss = model.loss(x)
                self.assertEqual(0, len(loss.shape))
                break  # only test the first batch


    def test_evaluate(self):
        for model in self.autoencoders:
            for x in self.train_loader:
                model.eval()
                true_x = torch.argmax(model.forward(x)[0], dim=1, keepdim=False)
                metrics = model.evaluate(x, true_x)
                self.assertEqual(metrics['n_samples'], x.nelement())
                self.assertEqual(metrics['correct'], x.nelement())

                metrics = model.evaluate(x, true_x * -1 + 3)
                self.assertEqual(metrics['correct'], 0)

                true_x[0, 0] = true_x[0, 0] * -1 + 3
                metrics = model.evaluate(x, true_x)
                self.assertEqual(metrics['correct'], x.nelement() - 1)

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
        self.assertEqual(nd_loss(x, z, self.ones).item(), bce_loss(z, x).item())

        nd_loss = NeighbourDistanceLoss(0.5)
        self.assertEqual(nd_loss(x, z, self.ones).item(), bce_loss(z, x).item() * 0.5)

        nd_loss = NeighbourDistanceLoss(1.0)
        self.assertEqual(nd_loss(x, z, self.ones).item(), 0.0)

        y = torch.arange(self.ones.nelement()).reshape(self.shape).float()
        self.assertEqual(nd_loss(x, z, y).item(), 1.0)


if __name__ == '__main__':
    unittest.main()
