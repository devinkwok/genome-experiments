# quick script for creating balanced datasets of TF factor snps

import torch
import numpy as np

def balance(x, labels, tf_column):
    tf_labels = labels[:, tf_column]
    positive = tf_labels[:, tf_column]
    b = [np.nonzero(a['labels'][:, x]) for x in range(919)]
    np.sort(b)
    np.nonzero(np.equal(b,tf_column))
