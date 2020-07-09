import sys
sys.path.append('./src/')
import argparse

import numpy as np
import torch

from autoencoder import *
from seq_util.io import output_path
from seq_util.datasets import SequenceDataset


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
    'output_len': 919,  # for supervised model
}


def load_data(model, dataset, split_prop, n_dataloader_workers):
    print("Split training and validation sets...")
    split_size = int(split_prop * len(dataset))
    if split_size == len(dataset):
        split_size = len(dataset) - 1
    train_data, valid_data = torch.utils.data.random_split(dataset, [len(dataset) - split_size, split_size])
    print("Create data loaders...")
    train_loader = torch.utils.data.DataLoader(train_data,
            batch_size=model.seq_per_batch, shuffle=True, num_workers=n_dataloader_workers)
    valid_loader = torch.utils.data.DataLoader(valid_data,
            batch_size=model.seq_per_batch*2, shuffle=False, num_workers=n_dataloader_workers)
    return train_loader, valid_loader


def evaluate_model(model, valid_loader, device, disable_eval):
    if not disable_eval:
        model.eval()
    with torch.no_grad():
        total_metrics = {}
        for sample in valid_loader:

            if len(sample) == 2:
                x, labels = sample[0].to(device), sample[1].to(device)
                del sample
                metrics = model.evaluate(x, labels)
            else:
                labels = None
                x = sample.to(device)
                del sample
                metrics = model.evaluate(x, x)

            del x, labels
            for key, value in metrics.items():
                if key in total_metrics:
                    total_metrics[key] += value
                else:
                    total_metrics[key] = value
    EPSILON = 0.0001  # add epsilon to avoid division by 0
    if 'true_pos' in total_metrics and 'true_neg' in total_metrics:
        total_metrics['correct'] = total_metrics['true_pos'] + total_metrics['true_neg']
    if 'correct' in total_metrics and 'n_samples' in total_metrics:
        total_metrics['accuracy'] = total_metrics['correct'] / total_metrics['n_samples']
    if 'true_pos' in total_metrics and 'false_pos' in total_metrics:
        total_metrics['precision'] = total_metrics['true_pos'] / (total_metrics['true_pos'] + total_metrics['false_pos'] + EPSILON)
    if 'true_pos' in total_metrics and 'false_neg' in total_metrics:
        total_metrics['recall'] = total_metrics['true_pos'] / (total_metrics['true_pos'] + total_metrics['false_neg'] + EPSILON)
    if 'precision' in total_metrics and 'recall' in total_metrics:
        total_metrics['f1'] = 2 * total_metrics['precision'] * total_metrics['recall'] / (total_metrics['precision'] + total_metrics['recall'] + EPSILON)
    return total_metrics


# disable_eval is a quick hack to turn evaluation code off and on, this is useful for testing
# model effectiveness while dropout and noise are included
def train(model, train_loader, valid_loader, optimizer, epochs, disable_eval, checkpoint_interval,
            use_cuda=False):
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
        for sample in train_loader:
            model.train()
            if len(sample) == 2:
                x, labels = sample[0].to(device), sample[1].to(device)
                del sample
                loss = model.loss(x, labels)
            else:
                labels = None
                x = sample.to(device)
                del sample
                loss = model.loss(x)
            loss_sum += loss.item()
            n_batches += 1
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            del x, loss, labels

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
        dataset = SequenceDataset(config['input_path'], seq_len=model.seq_len, overlap=(model.kernel_len - 1))

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
    for model, i, j, metrics in checkpoints:
        print("Model evaluation at epoch {}, batch {}, metrics {}".format(i, j, metrics))
        if config['save_model']:
            model_str = "{}{}x{}d{}n{}l{}_{}at{}_{}-{}".format(config['model'],
                    model.kernel_len, model.latent_len, model.input_dropout_freq,
                    model.latent_noise_std, config['neighbour_loss_prop'],
                    model.total_epochs + config['epochs'], config['learn_rate'], i, j)
            out_file = output_path(config['name'], config['input_path'], model_str + '.pth')
            print("Saving model to {}".format(out_file))
            torch.save(model.state_dict(), out_file)

    return model, train_loader, valid_loader, optimizer, loss_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Configuration and hyperparameters for autoencoder.')
    for key, value in __CONFIG_DEFAULT.items():
        parser.add_argument('--' + key, type=type(value), required=False, default=value)
    config = parser.parse_args()
    run(vars(config))
