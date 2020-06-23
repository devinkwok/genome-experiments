import sys
sys.path.append('./src/')
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import data_in
from datasets import SequenceDataset
import seq_util.io

N_BASE = data_in.N_BASE

# input tensors are (n, N_BASE, subseq_len)
# where n is number of subsequences per batch, N_BASE is number of channels
# and subseq_len is the length of subsequences

__CONFIG_DEFAULT = {
    'name': 'DEFAULT',
    'model': 'Autoencoder',
    'kernel_len': 1,
    'latent_len': 1,
    'seq_len': 1,
    'seq_per_batch': 1,
    'input_path': "",
    'split_prop': 0.05,
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
    'hidden_dropout_freq': 0.1,
    'fixed_random_seed': True,
    'n_dataloader_workers': 2,
    'checkpoint_interval': 1000,
    'output_len': 919,      # for supervised model
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
            return self.bce_loss(z, x) * (1 - self.neighbour_loss_prop) + \
                self.mse_loss(y[:, :, :-1], y[:, :, 1:]) * self.neighbour_loss_prop
        return self.bce_loss(z, x)


class Autoencoder(nn.Module):


    def __init__(self, kernel_len, latent_len, seq_len, seq_per_batch, input_dropout_freq, latent_noise_std, loss_fn):
        super().__init__()
        self.kernel_len = kernel_len
        self.latent_len = latent_len
        self.seq_len = seq_len
        self.seq_per_batch = seq_per_batch
        self.input_dropout_freq = input_dropout_freq
        self.latent_noise_std = latent_noise_std
        self.loss_fn = loss_fn

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


    def loss(self, x):
        reconstructed, latent = self.forward(x)
        return self.loss_fn(x, reconstructed, latent)


    def evaluate(self, x, true_x):
        z, y = self.forward(x)
        predictions = predict(z)
        correct = (data_in.one_hot_to_seq(true_x) == data_in.one_hot_to_seq(predictions))
        accuracy = torch.sum(correct).item() / correct.nelement()
        error_indexes = torch.nonzero(torch.logical_not(correct), as_tuple=True)
        return accuracy, error_indexes


# class DilationEncoder(Autoencoder):
#     pass

class MultilayerEncoder(Autoencoder):

    def __init__(self, kernel_len, latent_len, seq_len, seq_per_batch, input_dropout_freq, latent_noise_std, loss_fn,
                hidden_len, pool_size, n_conv_and_pool, n_conv_before_pool, n_linear, hidden_dropout_freq):
        
        super().__init__(kernel_len, latent_len, seq_len, seq_per_batch, input_dropout_freq, latent_noise_std, loss_fn)

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
            encode_layers['dropout{}'.format(i)] = nn.Dropout(hidden_dropout_freq)

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


    def encode(self, x, is_onehot=False):
        if not is_onehot:
            one_hot = F.one_hot(x, num_classes=N_BASE).permute(0, 2, 1).type(torch.float32)
        else:
            one_hot = x
        return super().encode(one_hot)


    # need to convert to one hot first
    def loss(self, x):
        one_hot = F.one_hot(x, num_classes=N_BASE).permute(0, 2, 1).type(torch.float32)
        latent = super().encode(one_hot)
        reconstructed = self.decode(latent)
        return self.loss_fn(one_hot, reconstructed, latent)


    # compare indexes instead of one-hot
    def evaluate(self, x, true_x):
        z, _ = self.forward(x)
        predictions = torch.argmax(z, 1, keepdim=False)
        correct = (true_x == predictions)
        accuracy = torch.sum(correct).item() / correct.nelement()
        error_indexes = torch.nonzero(torch.logical_not(correct), as_tuple=True)
        return accuracy, error_indexes


class LatentLinearRegression(Autoencoder):

    def __init__(self, kernel_len, latent_len, seq_len, seq_per_batch, input_dropout_freq, latent_noise_std, loss_fn,
                encoder, output_dim):
        super().__init__(kernel_len, latent_len, seq_len, seq_per_batch, input_dropout_freq, latent_noise_std, loss_fn)
        self.encoder = encoder
        self.linear = nn.Linear(latent_len, output_dim, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.loss_fn = nn.BCELoss()


    def forward(self, x):
        with torch.no_grad():
            latent = self.encoder.encode(x, is_onehot=True)
        y = self.linear(latent)
        return self.sigmoid(y)


    def loss(self, x, labels):
        y = self.forward(x)
        return self.loss_fn(y, labels.type(torch.float32))


    def evaluate(self, x, labels):
        y = self.forward(x)
        y_true = (y > 0.5)
        label_true = (labels > 0.5)
        y_false = torch.logical_not(y_true)
        label_false = torch.logical_not(label_true)
        true_pos = torch.sum(torch.logical_and(y_true, label_true)).item()
        true_neg = torch.sum(torch.logical_and(y_false, label_false)).item()
        false_pos = torch.sum(torch.logical_and(y_true, label_false)).item()
        false_neg = torch.sum(torch.logical_and(y_false, label_true)).item()
        return {'n_samples': y.nelement(),
                'true_pos': true_pos,
                'true_neg': true_neg,
                'false_pos': false_pos,
                'false_neg': false_neg}


class LabelledSequence(torch.utils.data.Dataset):

    def __init__(self, filename, input_seq_len):
        data = torch.load(filename)
        self.labels = torch.tensor(data['labels'][:, (891, 914)])
        self.one_hot = torch.tensor(data['x'][:, :, 500-int(input_seq_len/2):500+int(input_seq_len/2)])


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, index):
        return self.one_hot[index], self.labels[index]


# if no output is above the cutoff, predict empty (none)
def predict(reconstruction, empty_cutoff_prob=(1 / N_BASE)):
    probabilities, indexes = torch.max(reconstruction, 1, keepdim=False)
    indexes[(probabilities <= empty_cutoff_prob)] = N_BASE
    output = F.one_hot(indexes, num_classes=N_BASE + 1)
    output = output[:,:,:-1]
    return output.permute(0, 2, 1)


def load_data(model, dataset, split_prop, n_dataloader_workers):
    print("Split training and validation sets...")
    split_size = int(split_prop * len(dataset))
    train_data, valid_data = torch.utils.data.random_split(dataset, [len(dataset) - split_size, split_size])
    print("Create data loaders...")
    train_loader = torch.utils.data.DataLoader(train_data,
            batch_size=model.seq_per_batch, shuffle=True, num_workers=n_dataloader_workers)
    valid_loader = torch.utils.data.DataLoader(valid_data,
            batch_size=model.seq_per_batch, shuffle=True, num_workers=n_dataloader_workers)
    return train_loader, valid_loader


def evaluate_model(model, valid_loader, device, disable_eval):
    if not disable_eval:
        model.eval()
    with torch.no_grad():
        total_metrics = {}
        for sample, labels in valid_loader:
            x = sample.to(device)
            del sample
            metrics = model.evaluate(x, labels)
            del x
            for key, value in metrics.items():
                if key in total_metrics:
                    total_metrics[key] += value
                else:
                    total_metrics[key] = value
    total_metrics['accuracy'] = (total_metrics['true_pos'] + total_metrics['true_neg'])/ total_metrics['n_samples']
    total_metrics['precision'] = total_metrics['true_pos'] / (total_metrics['true_pos'] + total_metrics['false_pos'] + 0.001)
    total_metrics['recall'] = total_metrics['true_pos'] / (total_metrics['true_pos'] + total_metrics['false_neg'] + 0.001)
    total_metrics['f1'] = 2 * total_metrics['precision'] * total_metrics['recall'] / (total_metrics['precision'] + total_metrics['recall'] + 0.001)
    return total_metrics


# disable_eval is a quick hack to turn evaluation code off and on, this is useful for testing
# model effectiveness while dropout and noise are included
def train(model, train_loader, valid_loader, optimizer, epochs, disable_eval, checkpoint_interval, use_cuda=False):
    # target CPU or GPU
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print("Using device:")
    print(device)

    model.to(device)
    n_batches = 0
    for i in range(epochs):
        model.total_epochs += 1
        loss_sum = 0
        for sample, labels in train_loader:
            model.train()
            x = sample.to(device)
            del sample
            loss = model.loss(x, labels)
            loss_sum += loss.item()
            n_batches += 1
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            del x, loss

            if n_batches % checkpoint_interval == 0:
                avg_loss = loss_sum / checkpoint_interval
                loss_sum = 0
                metrics = evaluate_model(model, valid_loader, device, disable_eval)
                print("epoch {}, batch {}, avg loss {}, metrics {}".format(
                        i, n_batches, avg_loss, metrics))
                yield model, i, n_batches, metrics

    metrics = evaluate_model(model, valid_loader, device, disable_eval)
    yield model, epochs, len(train_loader), metrics


def run(update_config):
    config = dict(__CONFIG_DEFAULT)
    config.update(update_config)
    print("Config values:")
    print(config)

    if config['fixed_random_seed']:
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    loss_fn = NeighbourDistanceLoss(config['neighbour_loss_prop'])

    if config['model'] == 'Multilayer' or config['model'] == 'LatentLinearRegression':
        model = MultilayerEncoder(config['kernel_len'], config['latent_len'], config['seq_len'],
                config['seq_per_batch'], config['input_dropout_freq'], config['latent_noise_std'], loss_fn,
                config['hidden_len'], config['pool_size'], config['n_conv_and_pool'],
                config['n_conv_before_pool'], config['n_linear'], config['hidden_dropout_freq'])
        if config['model'] == 'Multilayer':
            dataset = SequenceDataset(config['input_path'], model.seq_len)
    else:
        model = Autoencoder(config['kernel_len'], config['latent_len'], config['seq_len'],
                config['seq_per_batch'], config['input_dropout_freq'], config['latent_noise_std'], loss_fn)
        dataset = data_in.SeqData.from_file(config['input_path'], seq_len=model.seq_len,
                overlap=(model.kernel_len - 1), do_cull=True, cull_threshold=0.99)

    if not (config['load_prev_model_state'] is None):
        print("Loading model...")
        model.load_state_dict(torch.load(config['load_prev_model_state'], map_location=torch.device('cpu')))
    if config['model'] == 'LatentLinearRegression':
        encoder = model
        model = LatentLinearRegression(config['kernel_len'], config['latent_len'], config['seq_len'],
                config['seq_per_batch'], config['input_dropout_freq'], config['latent_noise_std'], loss_fn,
                encoder, config['output_len'])
        dataset = LabelledSequence(config['input_path'], config['seq_len'])
    print("Model specification:")
    print(model)
    print("Loading data...")
    train_loader, valid_loader = load_data(model, dataset, 
            config['split_prop'], config['n_dataloader_workers'])
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learn_rate'])
    print("Training for {} epochs".format(config['epochs']))

    checkpoints = train(model, train_loader, valid_loader, optimizer, config['epochs'],
            config['disable_eval'], config['checkpoint_interval'], use_cuda=config['use_cuda_if_available'])
    for model, i, j, accuracy in checkpoints:
        if config['save_model']:
            model_str = "{}{}x{}d{}n{}l{}_{}at{}_{}-{}".format(config['model'],
                    model.kernel_len, model.latent_len, model.input_dropout_freq,
                    model.latent_noise_std, config['neighbour_loss_prop'],
                    model.total_epochs + config['epochs'], config['learn_rate'], i, j)
            out_file = seq_util.io.output_path(config['name'], config['input_path'], model_str + '.pth')
            print("Saving model to {}".format(out_file))
            torch.save(model.state_dict(), out_file)

    return model, train_loader, valid_loader, optimizer, loss_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Configuration and hyperparameters for autoencoder.')
    for key, value in __CONFIG_DEFAULT.items():
        parser.add_argument('--' + key, type=type(value), required=False, default=value)
    config = parser.parse_args()
    run(vars(config))
