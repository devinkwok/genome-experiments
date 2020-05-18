import unittest

import torch
import numpy as np
import numpy.testing as npt

from src.ae.autoencoder import *


class Test_Autoencoder(unittest.TestCase):


    def setUp(self):
        self.ae = Autoencoder(8, 32, 4)


    def load_data_from_fasta(self):
        data_loader = self.ae.load_data_from_fasta('test.fasta', 10, 10)
