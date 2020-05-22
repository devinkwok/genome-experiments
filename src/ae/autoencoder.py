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
                seq_len=2, seq_per_batch=1, empty_cutoff_prob=(1 / N_BASE)):
        super().__init__()
        self.window_len = window_len
        self.latent_len = latent_len
        self.seq_len = seq_len
        self.seq_per_batch = seq_per_batch
        self.encode_layer1 = nn.Conv1d(N_BASE, latent_len, window_len)
        # need ConvTranspose to do deconvolution, otherwise channels go to wrong dimension
        self.decode_layer1 = nn.ConvTranspose1d(latent_len, N_BASE, window_len)
        self.empty_cutoff_prob = empty_cutoff_prob
        self.total_epochs = 0
        # for convenience, length of the sequence which is decoded by window
        self.decodable_len = self.seq_len - (self.window_len - 1)
        self.train_loader = None
        self.valid_loader = None
        self.input_path = None  #TODO quick hack


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
        return reconstructed


    def load_data(self, data_filename, validation_split=0.2):
        dataset = data_in.SeqData.from_file(data_filename, seq_len=self.seq_len,
            overlap=(self.window_len - 1), do_cull=True, cull_threshold=0.05)
        train_data, valid_data = dataset.split(split_prop=validation_split, shuffle=True)
        self.train_loader = torch.utils.data.DataLoader(
                train_data, batch_size=self.seq_per_batch, shuffle=True, num_workers=4)
        self.valid_loader = torch.utils.data.DataLoader(
                valid_data, batch_size=self.seq_per_batch, shuffle=True, num_workers=4)
        self.input_path = data_filename


    def train(self, epochs=1, learn_rate=0.01):
        if self.train_loader is None or self.valid_loader is None:
            raise RuntimeError("No data loaded, run Autoencoder.load_data() first.")
        # target CPU or GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        optimizer = torch.optim.SGD(self.parameters(), lr=learn_rate)
        # TODO use BCE with logits as more stable than BCE with sigmoid separately?
        loss_fn = nn.MSELoss()
        validation = iter(self.valid_loader)

        model_str = "{}x{}_{}_at{}".format(
                self.window_len, self.latent_len, self.total_epochs + epochs, learn_rate)
        print("Training {} for {} epochs".format(model_str, epochs))
        for i in range(epochs):
            self.total_epochs += 1
            loss_sum = 0
            for x, true_x in self.train_loader:
                optimizer.zero_grad()
                z = self.forward(x)
                loss = loss_fn(z, true_x)
                loss_sum += loss.item()
                loss.backward()
                optimizer.step()

            accuracy, _ = self.evaluate(*next(validation))
            print("epoch {}, avg loss {}, validation acc. {}".format(
                    i, loss_sum / len(self.train_loader), accuracy))
        
        out_file = seq_util.io.output_path("ae01_", self.input_path, model_str + '.pth')
        print("Saving model to {}".format(out_file))
        torch.save(self, out_file)


    def predict(self, reconstruction):
        probabilities, indexes = torch.max(reconstruction, 1, keepdim=False)
        indexes[(probabilities <= self.empty_cutoff_prob)] = N_BASE
        output = F.one_hot(indexes, num_classes=N_BASE + 1)
        output = output[:,:,:-1]
        return output.permute(0, 2, 1)


    def evaluate(self, x, true_x):
        with torch.no_grad():
            z = self.forward(x)
            predictions = self.predict(z)
            correct = (data_in.one_hot_to_seq(true_x) == data_in.one_hot_to_seq(predictions))
            accuracy = torch.sum(correct).item() / correct.nelement()
            error_indexes = torch.nonzero(torch.logical_not(correct), as_tuple=True)
            #FIXME accuracy doesn't seem to be on each input separately
        return accuracy, error_indexes


    #TODO add noise (dropout)
    #TODO use gpu
    #TODO reserve some inputs for evaluateing


def experiment_1():
    input_path = "data/ref_genome/test.fasta"
    # one latent var isn't enough, >= 2 bits should capture all 4 bp
    ae = Autoencoder(window_len=1, latent_len=1, seq_len=32, seq_per_batch=4)
    ae.load_data(input_path)
    ae.train(epochs=20, learn_rate=0.01)

    ae = Autoencoder(window_len=1, latent_len=2, seq_len=32, seq_per_batch=4)
    ae.load_data(input_path)
    ae.train(epochs=20, learn_rate=0.01)

    ae = Autoencoder(window_len=1, latent_len=4, seq_len=32, seq_per_batch=4)
    ae.load_data(input_path)
    ae.train(epochs=20, learn_rate=0.01)

    # similar experiment on window size 3, >= 2 * 3 bits should capture all variation
    ae = Autoencoder(window_len=3, latent_len=2, seq_len=32, seq_per_batch=4)
    ae.load_data(input_path)
    ae.train(epochs=20, learn_rate=0.01)

    ae = Autoencoder(window_len=3, latent_len=4, seq_len=32, seq_per_batch=4)
    ae.load_data(input_path)
    ae.train(epochs=20, learn_rate=0.01)

    ae = Autoencoder(window_len=3, latent_len=6, seq_len=32, seq_per_batch=4)
    ae.load_data(input_path)
    ae.train(epochs=20, learn_rate=0.01)

    ae = Autoencoder(window_len=3, latent_len=12, seq_len=32, seq_per_batch=4)
    ae.load_data(input_path)
    ae.train(epochs=20, learn_rate=0.01)

    ae = Autoencoder(window_len=3, latent_len=24, seq_len=32, seq_per_batch=4)
    ae.load_data(input_path)
    ae.train(epochs=20, learn_rate=0.01)


if __name__ == '__main__':
    experiment_1()
    # ae = torch.load('outputs/src/ae/autoencoder/ae01_test1x4_20_at0.01.pth')
