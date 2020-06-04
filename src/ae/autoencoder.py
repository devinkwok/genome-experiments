import sys
sys.path.append('./src/')
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import data_in
import seq_util.io

N_BASE = data_in.N_BASE

# input tensors are (n, N_BASE, subseq_len)
# where n is number of subsequences per batch, N_BASE is number of channels
# and subseq_len is the length of subsequences

_DEFAULT_HYPERPARAMETERS = {
    'name': "default",
    'window_len': 1,
    'latent_len': 1,
    'seq_len': 1,
    'seq_per_batch': 1,
    'input_path': "",
    'split_prop': 0.0,
    'epochs': 0,
    'learn_rate': 0.01,
    'dropout_freq': 0.0,
    'noise_std': 0.0,
    'save_model': True,
    'disable_eval': False,
    'neighbour_loss_prop': 0.0,
    'load_prev_model_state': None,
}


# this layer drops out every channel at some positions, and keeps inputs at size 1
class SeqDropout(nn.Module):

    def __init__(self, dropout_freq):
        super().__init__()
        self.dropout = nn.Dropout2d(dropout_freq)

    def forward(self, x):
        y = self.dropout(x.permute(0, 2, 1)).permute(0, 2, 1)
        return y * (1 - self.dropout.p)


class GaussianNoise(nn.Module):

    def __init__(self, noise_std):
        super().__init__()
        self.noise_std = noise_std

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x)
            return x + noise * self.noise_std
        return x


# this loss function combines binary cross entropy with neighbour distance
# neighbour distance is difference between latent variables at adjacent positions
# the output loss is weighted between BCE loss and this difference
class NeighbourDistanceLoss(nn.Module):

    def __init__(self, neighbour_loss_prop):
        super().__init__()
        self.neighbour_loss_prop = neighbour_loss_prop
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, x, z, y):
        return self.bce_loss(x, z) * (1 - self.neighbour_loss_prop) + \
                self.mse_loss(y[:, :, :-1], y[:, :, 1:]) * self.neighbour_loss_prop


class Autoencoder(nn.Module):


    def __init__(self, window_len, latent_len, seq_len, seq_per_batch, dropout_freq, noise_std):
        super().__init__()
        self.window_len = window_len
        self.latent_len = latent_len
        self.seq_len = seq_len
        self.seq_per_batch = seq_per_batch
        self.dropout_freq = dropout_freq
        self.noise_std = noise_std

        self.total_epochs = 0  # tracks number of epochs this model has been trained

        self.encode = nn.Sequential(
            SeqDropout(self.dropout_freq),
            nn.Conv1d(N_BASE, latent_len, window_len),
            nn.ReLU(),
            )
        self.decode = nn.Sequential(
            GaussianNoise(self.noise_std),
            # need ConvTranspose to do deconvolution, otherwise channels go to wrong dimension
            nn.ConvTranspose1d(latent_len, N_BASE, window_len),
            nn.Softmax(dim=1),
            )


    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent


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
    z, y = model.forward(x)
    predictions = predict(z)
    correct = (data_in.one_hot_to_seq(true_x) == data_in.one_hot_to_seq(predictions))
    accuracy = torch.sum(correct).item() / correct.nelement()
    error_indexes = torch.nonzero(torch.logical_not(correct), as_tuple=True)
    #FIXME accuracy doesn't seem to be on each input separately
    return accuracy, error_indexes


# disable_eval is a quick hack to turn evaluation code off and on, this is useful for testing
# model effectiveness while dropout and noise are included
def train(model, train_loader, valid_loader, optimizer, loss_fn, epochs, disable_eval):
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
            z, y = model.forward(x)
            loss = loss_fn(z, true_x, y)
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()

        if not disable_eval:
            model.eval()
        accuracy, _ = evaluate(model, *next(validation))
        print("epoch {}, avg loss {}, validation acc. {}".format(
                i, loss_sum / len(train_loader), accuracy))


def run(hparams):
    config = dict(_DEFAULT_HYPERPARAMETERS)
    config.update(hparams)

    model = Autoencoder(config['window_len'], config['latent_len'], config['seq_len'],
            config['seq_per_batch'], config['dropout_freq'], config['noise_std'])
    if not config['load_prev_model_state'] is None:
        model.load_state_dict(torch.load(config['load_prev_model_state']))

    train_loader, valid_loader = load_data(model, config['input_path'], config['split_prop'])
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learn_rate'])
    loss_fn = NeighbourDistanceLoss(config['neighbour_loss_prop'])

    model_str = "{}x{}d{}n{}l{}_{}_at{}".format(
            model.window_len, model.latent_len, model.dropout_freq, model.noise_std,
            config['neighbour_loss_prop'],
            model.total_epochs + config['epochs'], config['learn_rate'])
    print("Training {} for {} epochs".format(model_str, config['epochs']))

    train(model, train_loader, valid_loader, optimizer, loss_fn,
            config['epochs'], config['disable_eval'])

    if config['save_model']:
        out_file = seq_util.io.output_path(config['name'], config['input_path'], model_str + '.pth')
        print("Saving model to {}".format(out_file))
        torch.save(model.state_dict(), out_file)

    return model, train_loader, valid_loader, optimizer, loss_fn
