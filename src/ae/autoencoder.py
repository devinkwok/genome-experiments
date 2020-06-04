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


# this layer drops out every channel at some positions, and keeps inputs at size 1
class SeqDropout(nn.Module):

    def __init__(self, dropout_freq):
        super().__init__()
        self.dropout = nn.Dropout2d(dropout_freq)

    def forward(self, x):
        y = self.dropout(x.permute(0, 2, 1)).permute(0, 2, 1)
        return y * (1 - self.dropout.p)


class Autoencoder(nn.Module):


    def __init__(self, window_len=1, latent_len=1,
                seq_len=2, seq_per_batch=1, dropout_freq=0.0):
        super().__init__()
        self.window_len = window_len
        self.latent_len = latent_len
        self.seq_len = seq_len
        self.seq_per_batch = seq_per_batch
        self.dropout_freq = dropout_freq

        self.total_epochs = 0

        encode_layers = []
        if self.dropout_freq > 0:
            encode_layers.append(SeqDropout(self.dropout_freq))
        encode_layers.append(nn.Conv1d(N_BASE, latent_len, window_len))
        encode_layers.append(nn.ReLU())
        self.encode = nn.Sequential(*encode_layers)

        decode_layers = []
        # need ConvTranspose to do deconvolution, otherwise channels go to wrong dimension
        decode_layers.append(nn.ConvTranspose1d(latent_len, N_BASE, window_len))
        decode_layers.append(nn.Softmax(dim=1))
        self.decode = nn.Sequential(*decode_layers)


    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed


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


# if no output is above the cutoff, predict empty (none)
def predict(reconstruction, empty_cutoff_prob=(1 / N_BASE)):
    probabilities, indexes = torch.max(reconstruction, 1, keepdim=False)
    indexes[(probabilities <= empty_cutoff_prob)] = N_BASE
    output = F.one_hot(indexes, num_classes=N_BASE + 1)
    output = output[:,:,:-1]
    return output.permute(0, 2, 1)


def evaluate(model, x, true_x):
    z = model.forward(x)
    predictions = predict(z)
    correct = (data_in.one_hot_to_seq(true_x) == data_in.one_hot_to_seq(predictions))
    accuracy = torch.sum(correct).item() / correct.nelement()
    error_indexes = torch.nonzero(torch.logical_not(correct), as_tuple=True)
    #FIXME accuracy doesn't seem to be on each input separately
    return accuracy, error_indexes


def train(model, train_loader, valid_loader, optimizer, loss_fn, epochs):
    # target CPU or GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    validation = iter(valid_loader)

    for i in range(epochs):
        model.train()
        model.total_epochs += 1
        loss_sum = 0
        for x, true_x in train_loader:
            optimizer.zero_grad()
            z = model.forward(x)
            loss = loss_fn(z, true_x)
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()

        model.eval()
        accuracy, _ = evaluate(model, *next(validation))
        print("epoch {}, avg loss {}, validation acc. {}".format(
                i, loss_sum / len(train_loader), accuracy))


def run(hparams):
    model = Autoencoder(hparams['window_len'], hparams['latent_len'], hparams['seq_len'],
            hparams['seq_per_batch'], hparams['dropout_freq'])
    train_loader, valid_loader = load_data(model, hparams['input_path'], hparams['split_prop'])

    model_str = "{}x{}_drop{}_{}_at{}".format(
            model.window_len, model.latent_len, model.dropout_freq,
            model.total_epochs + hparams['epochs'], hparams['learn_rate'])
    print("Training {} for {} epochs".format(model_str, hparams['epochs']))

    optimizer = torch.optim.SGD(model.parameters(), lr=hparams['learn_rate'])
    # TODO use BCE with logits as more stable than BCE with sigmoid separately?
    loss_fn = nn.MSELoss()
    train(model, train_loader, valid_loader, optimizer, loss_fn, hparams['epochs'])

    out_file = seq_util.io.output_path(hparams['name'], hparams['input_path'], model_str + '.pth')
    print("Saving model to {}".format(out_file))
    torch.save(model.state_dict(), out_file)


def experiment_1():
    hparams = {
        'name': "ae01drop",
        'window_len': 1,
        'latent_len': 1,
        'seq_len': 32,
        'seq_per_batch': 4,
        'input_path': "data/ref_genome/test.fasta",
        'split_prop': 0.2,
        'epochs': 20,
        'learn_rate': 0.01,
        'dropout_freq': 0.2,
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
