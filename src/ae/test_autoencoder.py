import sys
sys.path.append('./src/ae/')
sys.path.append('./src/seq_util/')

import math
import unittest

import torch
import torch.nn as nn
import numpy as np
import numpy.testing as npt

from train import update_config, load_model, get_dataloaders
from autoencoder import *
from datasets import SequenceDataset

class Test_Autoencoder(unittest.TestCase):


    def setUp(self):
        config = {
            'model': 'Multilayer',
            'kernel_len': 5,
            'latent_len': 2,
            'seq_len': 20,
            'seq_per_batch': 7,
            'input_path': "data/ref_genome/test.fasta",
            'hidden_len': 10,
            'pool_size': 2,
            'n_conv_and_pool': 2,
            'n_conv_before_pool': 2,
            'n_linear': 2,
            'hidden_dropout_freq': 0.0,
        }
        self.config = update_config(config)
        self.autoencoders = []
        self.autoencoders.append(load_model(self.config))
        self.config['model'] = 'Autoencoder'
        self.autoencoders.append(load_model(self.config))
        self.train_loader, self.valid_loader = get_dataloaders(self.config)
        self.shape = (5, 4, 3)
        self.ones = torch.ones(*self.shape)
        self.zeros = torch.zeros(*self.shape)


    def test_load_data(self):
        for x in self.train_loader:
            self.assertEqual(x.shape, (self.config['seq_per_batch'], self.config['seq_len']))
            break  # only test the first batch
        for x in self.valid_loader:
            self.assertEqual(x.shape, (self.config['seq_per_batch'] * 2, self.config['seq_len']))
            break  # only test the first batch


    def test_forward(self):
        for model in self.autoencoders:
            for x in self.train_loader:
                reconstructed, latent = model.forward(x)
                latent_shape = (model.seq_per_batch, model.latent_len, model.seq_len - (model.kernel_len - 1))
                #TODO need Multilayer encoder to be fully convolutional during training
                # self.assertEqual(latent.shape, latent_shape)
                self.assertEqual(reconstructed.shape, (x.shape[0], 4, x.shape[1]))
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


    def test_decapitate(self):
        for model in self.autoencoders:
            keep_n_layers = 1
            model.decapitate(keep_n_layers=keep_n_layers)
            self.assertEqual(len(model.encode_layers), keep_n_layers)
            self.assertEqual(len(model.decode_layers), 0)
            for x in self.train_loader:
                y = model.encode(x)
                z = model.decode(y, softmax=False)
                npt.assert_array_equal(y, z)
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


    def test_ReverseComplementConv1d(self):
        with torch.no_grad():
            in_channels, out_channels, kernel_len = 4, 2, 6
            x = torch.randn(12, in_channels, 20)
            conv = ReverseComplementConv1d(in_channels, out_channels, 0, 0, 0, kernel_len)
            self.assertEqual(conv(x).shape, nn.Conv1d(in_channels, out_channels, kernel_len)(x).shape)
            npt.assert_array_equal(conv(x), conv.convs['normal'](x))

            def test_complement(x, conv):
                x_complement = torch.flip(x, [1])
                npt.assert_array_equal(conv(x), conv(x_complement))
            test_complement(x, ReverseComplementConv1d(in_channels, 0, out_channels, 0, 0, kernel_len))

            def test_reverse(x, conv):
                y = conv(x + torch.flip(x, [2]))
                npt.assert_array_equal(y, torch.flip(y, [2]))
            test_reverse(x, ReverseComplementConv1d(in_channels, 0, 0, out_channels, 0, kernel_len))

            conv = ReverseComplementConv1d(in_channels, 0, 0, 0, out_channels, kernel_len)
            test_reverse(x, conv)
            test_complement(x, conv)

            conv = ReverseComplementConv1d(in_channels, out_channels, out_channels, out_channels, out_channels, kernel_len)
            target_shape = list(conv.convs['normal'](x).shape)
            target_shape[1] = target_shape[1] * 4
            self.assertEqual(list(conv(x).shape), target_shape)
            self.assertEqual(conv.out_channels, target_shape[1])

            conv = ReverseComplementConv1d(in_channels, 0, 0, 0, out_channels, kernel_len, max_pool=False)
            target_shape = list(conv.convs['reverse_complement'](x).shape)
            target_shape[1] = target_shape[1] * 4
            self.assertEqual(list(conv(x).shape), target_shape)
            self.assertEqual(conv.out_channels, target_shape[1])


if __name__ == '__main__':
    unittest.main()
