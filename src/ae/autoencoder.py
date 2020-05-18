import torch.nn as nn

import seq_io


class Autoencoder(nn.Module):

    def __init__(self, window_size, batch_len, n_batch):
        super(Model, self).__init__()
        self.window_size = window_size
        self.batch_len = batch_len
        self.n_batch = n_batch
        # self.hidden_layers = nn.Conv1d()

    #TODO define layers
        #each unit is linear, relu

    #TODO define forward pass
        # add noise (dropout)
        #TODO input to latent
        #TODO latent to reconstruction
            #TODO use softmax for reconstruction
    
    #TODO use gpu

    #TODO optimization code

    #TODO slice input into random pieces with overlap
        # reserve some for testing


    def load_data_from_fasta(self, filename, window_size, empty_freq):
        bio_seq = seq_io.read_seq(filename)
        seq1h_tensor = seq_io.seq_to_1h(bio_seq)
        augmented_seq = torch.cat((seq1h_tensor, seq_io.reverse(seq1h_tensor)), 0)
        augmented_seq = torch.cat((augmented_seq, seq_io.complement(augmented_seq)), 0)
        input_data = seq_io.slice_seq(augmented_seq, self.batch_len, self.window_size)
        input_data = seq_io.cull_empty(input_data, base_freq=empty_freq)
        return torch.utils.data.DataLoader(
                input_data, batch_size=self.n_batch, shuffle=True, num_workers=4)
