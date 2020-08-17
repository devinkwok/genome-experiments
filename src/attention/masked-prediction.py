
import torch.nn as nn


class MaskedPredictionTask(nn.Module):


    def __init__(self):
        pass


    def forward(self, x):
        # choose non-empty positions in x to predict as x'
        # mask a proportion of x'
        # randomly replace a proportion of x'
        # leave a proportion of x' unchanged
        # run encoder and decoder to generate predictions from x
        # loss function comparing predicted x' to x'