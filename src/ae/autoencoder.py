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
# where n is number of subsequences per batch, N_BASE is number of channels
# and subseq_len is the length of subsequences

class Autoencoder(nn.Module):


    def __init__(self, window_len=1, latent_len=1,
                seq_len=2, seq_per_batch=1, empty_cutoff_prob=(1 / N_BASE)):
        super().__init__()
        self.window_len = window_len
        self.latent_len = latent_len
        self.seq_len = seq_len
        self.seq_per_batch = seq_per_batch
        self.empty_cutoff_prob = empty_cutoff_prob  # if no output is above this cutoff, predict empty (none)
        self.encode_layer1 = nn.Conv1d(N_BASE, latent_len, window_len)
        # need ConvTranspose to do deconvolution, otherwise channels go to wrong dimension
        self.decode_layer1 = nn.ConvTranspose1d(latent_len, N_BASE, window_len)
        self.total_epochs = 0
        # for convenience, length of the sequence which is decoded by window
        self.decodable_len = self.seq_len - (self.window_len - 1)


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


def load_data(model, input_path, split_prop):
    dataset = data_in.SeqData.from_file(input_path, seq_len=model.seq_len,
        overlap=(model.window_len - 1), do_cull=True, cull_threshold=0.99)
    split_size = int(split_prop * len(dataset))
    train_data, valid_data = torch.utils.data.random_split(dataset, [len(dataset) - split_size, split_size])
    train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=model.seq_per_batch, shuffle=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(
            valid_data, batch_size=model.seq_per_batch, shuffle=True, num_workers=4)
    return train_loader, valid_loader


def train(model, train_loader, valid_loader, optimizer, loss_fn, epochs):
    # target CPU or GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    validation = iter(valid_loader)

    for i in range(epochs):
        model.total_epochs += 1
        loss_sum = 0
        for x, true_x in train_loader:
            optimizer.zero_grad()
            z = model.forward(x)
            loss = loss_fn(z, true_x)
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()

        accuracy, _ = model.evaluate(*next(validation))
        print("epoch {}, avg loss {}, validation acc. {}".format(
                i, loss_sum / len(train_loader), accuracy))


def run(hparams):
    model = Autoencoder(hparams['window_len'], hparams['latent_len'],
                hparams['seq_len'], hparams['seq_per_batch'], hparams['empty_cutoff_prob'])
    train_loader, valid_loader = load_data(model, hparams['input_path'], hparams['split_prop'])

    model_str = "{}x{}_{}_at{}".format(
            model.window_len, model.latent_len,
            model.total_epochs + hparams['epochs'], hparams['learn_rate'])
    print("Training {} for {} epochs".format(model_str, hparams['epochs']))

    optimizer = torch.optim.SGD(model.parameters(), lr=hparams['learn_rate'])
    # TODO use BCE with logits as more stable than BCE with sigmoid separately?
    loss_fn = nn.MSELoss()
    train(model, train_loader, valid_loader, optimizer, loss_fn, hparams['epochs'])

    out_file = seq_util.io.output_path("ae01_", hparams['input_path'], model_str + '.pth')
    print("Saving model to {}".format(out_file))
    torch.save(model.state_dict(), out_file)


def experiment_1():
    hparams = {
        'window_len': 1,
        'latent_len': 1,
        'seq_len': 32,
        'seq_per_batch': 4,
        'empty_cutoff_prob': 0.25,
        'input_path': "data/ref_genome/test.fasta",
        'split_prop': 0.2,
        'epochs': 20,
        'learn_rate': 0.01,
    }
    # one latent var isn't enough, >= 2 bits should capture all 4 bp
    run(hparams)

    hparams['latent_len'] = 2
    run(hparams)

    hparams['latent_len'] = 4
    run(hparams)

    # similar experiment on window size 3, >= 2 * 3 bits should capture all variation
    hparams['window_len'] = 3
    hparams['latent_len'] = 2
    run(hparams)

    hparams['latent_len'] = 4
    run(hparams)

    hparams['latent_len'] = 6
    run(hparams)

    hparams['latent_len'] = 12
    run(hparams)

    hparams['latent_len'] = 24
    run(hparams)


if __name__ == '__main__':
    experiment_1()
    # ae = torch.load('outputs/src/ae/autoencoder/ae01_test1x4_20_at0.01.pth')
