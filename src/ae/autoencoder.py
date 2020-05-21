import sys
sys.path.append('./src/')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import data_in
import seq_util.io

debug = True
N_BASE = data_in.N_BASE

# for reproducibility
if debug:
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# input tensors are (n, N_BASE, subseq_len)
# where n is number of batches, N_BASE is number of channels
# and subseq_len is the length of subsequences

class Autoencoder(nn.Module):

    def __init__(self, window_len=1, latent_len=1,
                seq_len=2, seq_per_batch=1):
        super().__init__()
        self.window_len = window_len
        self.latent_len = latent_len
        self.seq_len = seq_len
        self.seq_per_batch = seq_per_batch
        self.encode_layer1 = nn.Conv1d(N_BASE, latent_len, window_len)
        # need ConvTranspose to do deconvolution, otherwise channels go to wrong dimension
        self.decode_layer1 = nn.ConvTranspose1d(latent_len, N_BASE, window_len)
        self.total_epochs = 0
        # for convenience, length of the sequence which is decoded by window
        self.decodable_len = self.seq_len - self.window_len + 1
    
    def encode(self, x):
        y = self.encode_layer1(x)
        y = F.relu(y)
        return y
    
    def decode(self, y):
        z = self.decode_layer1(y)
        z = F.softmax(z, dim=1)
        return z

    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return latent, reconstructed

    #TODO add noise (dropout)
    #TODO use gpu
    #TODO reserve some inputs for testing


    def load_data(self, filename, cull_empty=False, cull_threshold=0.05):
        bio_seq = data_in.read_seq(filename)
        seq1h_tensor = data_in.seq_to_1h(bio_seq)

        # augment data with reverse and complement
        augmented_seq = torch.cat((seq1h_tensor, data_in.reverse(seq1h_tensor)), 0)
        augmented_seq = torch.cat((augmented_seq, data_in.complement(augmented_seq)), 0)

        # use overlap of (window size - 1) to ensure every position has convolution applied once
        input_data = data_in.slice_seq(augmented_seq, length=self.seq_len, overlap=(self.window_len - 1))
        if cull_empty:
            input_data = data_in.cull_empty(input_data, base_freq=cull_threshold)
        # TODO: this is a quick fix to get dimensions in correct order for Conv1D
        input_data = input_data.permute(0, 2, 1)

        return torch.utils.data.DataLoader(
                input_data, batch_size=self.seq_per_batch, shuffle=False, num_workers=4)


    def train(self, data_filename, epochs=1, learn_rate=0.01):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        data_loader = self.load_data("data/ref_genome/test.fasta")
        optimizer = torch.optim.SGD(self.parameters(), lr=learn_rate)
        # TODO use BCE with logits as more stable than BCE with sigmoid separately?
        loss_fn = nn.MSELoss()

        model_str = "{}x{}_{}_at{}".format(
                self.window_len, self.latent_len, self.total_epochs + epochs, learn_rate)
        print("Training {} for {} epochs".format(model_str, epochs))

        for i in range(epochs):
            self.total_epochs += 1
            loss_sum = 0

            for x in data_loader:
                optimizer.zero_grad()
                y, z = self(x)
                loss = loss_fn(z, x)
                loss_sum += loss.item()
                loss.backward()
                optimizer.step()
            
            print("epoch {}, average loss {}".format(i, loss_sum / len(data_loader)))
        
        out_file = seq_util.io.output_path("ae01_", input_path, model_str + '.pth')
        print("Saving model to {}".format(out_file))
        torch.save(ae, out_file)


if __name__ == '__main__':
    input_path = "data/ref_genome/test.fasta"

    # one latent var isn't enough, >= 2 bits should capture all 4 bp
    ae = Autoencoder(window_len=1, latent_len=1, seq_len=32, seq_per_batch=4)
    ae.train(input_path, epochs=20, learn_rate=0.01)

    ae = Autoencoder(window_len=1, latent_len=2, seq_len=32, seq_per_batch=4)
    ae.train(input_path, epochs=20, learn_rate=0.01)

    ae = Autoencoder(window_len=1, latent_len=4, seq_len=32, seq_per_batch=4)
    ae.train(input_path, epochs=20, learn_rate=0.01)

    # similar experiment on window size 3, >= 2 * 3 bits should capture all variation
    ae = Autoencoder(window_len=3, latent_len=2, seq_len=32, seq_per_batch=4)
    ae.train(input_path, epochs=20, learn_rate=0.01)

    ae = Autoencoder(window_len=3, latent_len=4, seq_len=32, seq_per_batch=4)
    ae.train(input_path, epochs=20, learn_rate=0.01)

    ae = Autoencoder(window_len=3, latent_len=6, seq_len=32, seq_per_batch=4)
    ae.train(input_path, epochs=20, learn_rate=0.01)

    ae = Autoencoder(window_len=3, latent_len=12, seq_len=32, seq_per_batch=4)
    ae.train(input_path, epochs=20, learn_rate=0.01)

    ae = Autoencoder(window_len=3, latent_len=24, seq_len=32, seq_per_batch=4)
    ae.train(input_path, epochs=20, learn_rate=0.01)
