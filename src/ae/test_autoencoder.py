import sys
sys.path.append('./src/ae/')

import unittest

import torch
import numpy as np
import numpy.testing as npt

import data_in
from autoencoder import *

class Test_Autoencoder(unittest.TestCase):


    def setUp(self):
        self.ae = Autoencoder(8, 3, 32, 4)
        pass


    def test_load_data_from_fasta(self):
        data_loader = self.ae.load_data_from_fasta("data/ref_genome/test.fasta")
        #TODO clean up

    def test_forward(self):
        #TODO clean up
        # test_np = np.array([
        #         [1, 0, 0, 0], [1, 0, 0, 0],
        #         [0, 1, 0, 0], [0, 1, 0, 0],
        #         [0, 0, 1, 0], [0, 0, 1, 0],
        #         [0, 0, 0, 1], [0, 0, 0, 1],
        #         [0, 0, 0, 0], [0, 0, 0, 0],
        #         ], dtype=np.float32)
        # test_tensor = torch.tensor(test_np).reshape((2, 5, 4))
        # test_tensor = test_tensor.permute(0, 2, 1)


        # print(test_tensor)
        # print(ae.forward(test_tensor))
        # print(ae.encode_layer1.weight)
        # print(ae.encode_layer1.bias)
        # for a in data_loader:
        #     print(a)
        #     print(ae.forward(a))
        #     break
        pass


if __name__ == '__main__':
    unittest.main()