import sys
sys.path.append('./src/')
from collections import OrderedDict

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
    'name': 'DEFAULT',
    'model': 'Autoencoder',
    'kernel_len': 1,
    'latent_len': 1,
    'seq_len': 1,
    'seq_per_batch': 1,
    'input_path': "",
    'split_prop': 0.0,
    'epochs': 0,
    'learn_rate': 0.01,
    'input_dropout_freq': 0.0,
    'latent_noise_std': 0.0,
    'save_model': True,
    'disable_eval': False,
    'neighbour_loss_prop': 0.0,
    'load_prev_model_state': None,
    'hidden_len': 1,
    'pool_size': 1,
    'n_conv_and_pool': 1,
    'n_conv_before_pool': 1,
    'n_linear': 1,
    'use_cuda_if_available': True,
}


class View(nn.Module):

    def __init__(self, target_shape):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, x):
        return x.view(self.target_shape)


# this layer drops out every channel at some positions, and keeps inputs at size 1
class SeqDropout(nn.Module):

    def __init__(self, input_dropout_freq):
        super().__init__()
        self.dropout = nn.Dropout2d(input_dropout_freq)

    def forward(self, x):
        y = self.dropout(x.permute(0, 2, 1)).permute(0, 2, 1)
        return y * (1 - self.dropout.p)


class GaussianNoise(nn.Module):

    def __init__(self, latent_noise_std):
        super().__init__()
        self.latent_noise_std = latent_noise_std

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x)
            return x + noise * self.latent_noise_std
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
        if self.neighbour_loss_prop > 0.0:
            return self.bce_loss(x, z) * (1 - self.neighbour_loss_prop) + \
                self.mse_loss(y[:, :, :-1], y[:, :, 1:]) * self.neighbour_loss_prop
        return self.bce_loss(x, z)


class Autoencoder(nn.Module):


    def __init__(self, kernel_len, latent_len, seq_len, seq_per_batch, input_dropout_freq, latent_noise_std):
        super().__init__()
        self.kernel_len = kernel_len
        self.latent_len = latent_len
        self.seq_len = seq_len
        self.seq_per_batch = seq_per_batch
        self.input_dropout_freq = input_dropout_freq
        self.latent_noise_std = latent_noise_std
        self.seq_overlap = self.kernel_len - 1

        self.total_epochs = 0  # tracks number of epochs this model has been trained

        self.encode_layers = nn.ModuleDict()
        self.encode_layers['input_dropout'] = SeqDropout(input_dropout_freq)
        self.encode_layers['conv0'] = nn.Conv1d(N_BASE, latent_len, kernel_len)
        self.encode_layers['relu0'] = nn.ReLU()

        self.decode_layers = nn.ModuleDict()
        self.decode_layers['latent_noise'] = GaussianNoise(self.latent_noise_std)
        self.decode_layers['conv0'] = nn.ConvTranspose1d(latent_len, N_BASE, kernel_len)
        self.decode_layers['softmax'] = nn.Softmax(dim=1)


    def encode(self, x):
        for layer in self.encode_layers.values():
            x = layer(x)
        return x


    def decode(self, y):
        for layer in self.decode_layers.values():
            y = layer(y)
        return y


    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent


# class DilationEncoder(Autoencoder):
#     pass

class MultilayerEncoder(Autoencoder):

    def __init__(self, kernel_len, latent_len, seq_len, seq_per_batch, input_dropout_freq, latent_noise_std,
                hidden_len, pool_size, n_conv_and_pool, n_conv_before_pool, n_linear):
        
        super().__init__(kernel_len, latent_len, seq_len, seq_per_batch, input_dropout_freq, latent_noise_std)
        self.seq_overlap = 0

        pad = int(kernel_len / 2)
        sizes = [N_BASE] + [hidden_len * (i + 1) for i in range(n_conv_and_pool)]
        in_size = sizes[:-1]
        out_size = sizes[1:]

        encode_layers = nn.ModuleDict()
        encode_layers['input_dropout'] = SeqDropout(input_dropout_freq)
        for i, (n_in, n_out) in enumerate(zip(in_size, out_size)):
            encode_layers['conv{}0'.format(i)] = nn.Conv1d(
                    n_in, n_out, kernel_len, 1, pad, padding_mode='zeros')
            encode_layers['relu{}0'.format(i)] = nn.ReLU()
            for j in range(1, n_conv_before_pool):
                encode_layers['conv{}{}'.format(i, j)] = nn.Conv1d(
                        n_out, n_out, kernel_len, 1, pad, padding_mode='zeros')
                encode_layers['relu{}{}'.format(i, j)] = nn.ReLU()
            encode_layers['pool{}'.format(i)] = nn.MaxPool1d(pool_size)
            encode_layers['norm{}'.format(i)] = nn.BatchNorm1d(n_out)
            encode_layers['dropout{}'.format(i)] = nn.Dropout(input_dropout_freq)

        linear_size = int(seq_len / (pool_size ** n_conv_and_pool))
        encode_layers['view'] = View((-1, linear_size * out_size[-1]))
        encode_layers['linear0'] = nn.Linear(linear_size * out_size[-1], latent_len)
        for i in range(1, n_linear):
            encode_layers['reluL{}'.format(i)] = nn.ReLU()
            encode_layers['normL{}'.format(i)] = nn.BatchNorm1d(latent_len)
            encode_layers['linear{}'.format(i)] = nn.Linear(latent_len, latent_len)

        decode_layers = nn.ModuleDict()
        decode_layers['latent_noise'] = GaussianNoise(latent_noise_std)
        for i in range(1, n_linear):
            decode_layers['linear{}'.format(i)] = nn.Linear(latent_len, latent_len)
            decode_layers['reluL{}'.format(i)] = nn.ReLU()
            decode_layers['normL{}'.format(i)] = nn.BatchNorm1d(latent_len)
        decode_layers['linear0'] = nn.Linear(latent_len, linear_size * out_size[-1])
        decode_layers['view'] = View((-1, out_size[-1], linear_size))

        for i, (n_in, n_out) in enumerate(zip(reversed(in_size), reversed(out_size))):
            decode_layers['relu{}0'.format(i)] = nn.ReLU()
            decode_layers['norm{}'.format(i)] = nn.BatchNorm1d(n_out)
            decode_layers['pool{}'.format(i)] = nn.Upsample(scale_factor=pool_size,
                    mode='linear', align_corners=False)
            for j in reversed(range(1, n_conv_before_pool)):
                decode_layers['conv{}{}'.format(i, j)] = nn.ConvTranspose1d(
                        n_out, n_out, kernel_len, 1, pad)
                decode_layers['relu{}{}'.format(i, j)] = nn.ReLU()
            decode_layers['conv{}0'.format(i)] = nn.ConvTranspose1d(
                    n_out, n_in, kernel_len, 1, pad)
        decode_layers['softmax'] = nn.Softmax(dim=1)

        self.encode_layers = encode_layers
        self.decode_layers = decode_layers


def load_data(model, input_path, split_prop):
    dataset = data_in.SeqData.from_file(input_path, seq_len=model.seq_len,
        overlap=model.seq_overlap, do_cull=True, cull_threshold=0.99)
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
    if config['use_cuda_if_available'] and torch.cuda.is_available():
        torch.cuda.set_device('cuda')

    if config['model'] == 'Multilayer':
        model = MultilayerEncoder(config['kernel_len'], config['latent_len'], config['seq_len'],
                config['seq_per_batch'], config['input_dropout_freq'], config['latent_noise_std'],
                config['hidden_len'], config['pool_size'], config['n_conv_and_pool'],
                config['n_conv_before_pool'], config['n_linear'])
    else:
        model = Autoencoder(config['kernel_len'], config['latent_len'], config['seq_len'],
                config['seq_per_batch'], config['input_dropout_freq'], config['latent_noise_std'])

    if not config['load_prev_model_state'] is None:
        model.load_state_dict(torch.load(config['load_prev_model_state']))
    train_loader, valid_loader = load_data(model, config['input_path'], config['split_prop'])
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learn_rate'])
    loss_fn = NeighbourDistanceLoss(config['neighbour_loss_prop'])

    model_str = "{}{}x{}d{}n{}l{}_{}at{}".format(config['model'],
            model.kernel_len, model.latent_len, model.input_dropout_freq,
            model.latent_noise_std, config['neighbour_loss_prop'],
            model.total_epochs + config['epochs'], config['learn_rate'])
    print("Model specification:")
    print(model)
    print("Config values:")
    print(config)
    print("Training for {} epochs".format(config['epochs']))

    train(model, train_loader, valid_loader, optimizer, loss_fn,
            config['epochs'], config['disable_eval'])

    if config['save_model']:
        out_file = seq_util.io.output_path(config['name'], config['input_path'], model_str + '.pth')
        print("Saving model to {}".format(out_file))
        torch.save(model.state_dict(), out_file)

    return model, train_loader, valid_loader, optimizer, loss_fn
