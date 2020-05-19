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

    def __init__(self, window_size, latent_size, batch_len, n_batch):
        super().__init__()
        self.window_size = window_size
        self.latent_size = latent_size
        self.batch_len = batch_len
        self.n_batch = n_batch
        self.empty_freq = 0.05
        # FIXME wrong sizes
        self.encode_layer1 = nn.Conv1d(N_BASE, latent_size, window_size)
        self.decode_layer1 = nn.Conv1d(latent_size, N_BASE, window_size)
        self.total_epochs = 0
    
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


    def load_data_from_fasta(self, filename):
        bio_seq = data_in.read_seq(filename)
        seq1h_tensor = data_in.seq_to_1h(bio_seq)
        augmented_seq = torch.cat((seq1h_tensor, data_in.reverse(seq1h_tensor)), 0)
        augmented_seq = torch.cat((augmented_seq, data_in.complement(augmented_seq)), 0)
        input_data = data_in.slice_seq(augmented_seq, self.batch_len, self.window_size)
        input_data = data_in.cull_empty(input_data, base_freq=self.empty_freq)
        # TODO: this is a quick fix to get dimensions in correct order for Conv1D
        input_data = input_data.permute(0, 2, 1)
        return torch.utils.data.DataLoader(
                input_data, batch_size=self.n_batch, shuffle=False, num_workers=4)

    def train(self, data_filename, epochs, learn_rate):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        data_loader = self.load_data_from_fasta("data/ref_genome/test.fasta")
        optimizer = torch.optim.SGD(self.parameters(), lr=learn_rate)
        # TODO use BCE with logits as more stable than BCE with sigmoid separately?
        loss_fn = nn.MSELoss()

        model_str = "{}x{}_{}_at{}".format(
                self.window_size, self.latent_size, self.total_epochs + epochs, learn_rate)
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
        
        out_file = seq_util.io.output_path("ae01_", input_path, model_str)
        print("Saving model to {}".format(out_file))
        torch.save(ae, out_file)


if __name__ == '__main__':
    input_path = "data/ref_genome/test.fasta"

    # one latent var isn't enough
    ae = Autoencoder(1, 1, 32, 4)
    ae.train(input_path, 20, 0.01)

    # 2 latent vars can theoretically capture 4 possible base pairs
    ae = Autoencoder(1, 2, 32, 4)
    ae.train(input_path, 20, 0.01)

    # 4 latent vars should capture all bases
    ae = Autoencoder(1, 4, 32, 4)
    ae.train(input_path, 20, 0.01)

    # window size 3
    ae = Autoencoder(3, 1, 32, 4)
    ae.train(input_path, 40, 0.01)

    ae = Autoencoder(3, 4, 32, 4)
    ae.train(input_path, 40, 0.01)

    ae = Autoencoder(3, 16, 32, 4)
    ae.train(input_path, 40, 0.01)
