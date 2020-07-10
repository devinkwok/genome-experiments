import sys
sys.path.append('./src/')
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from ae.autoencoder import *
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
    'log_path': 'logs',
    'snapshot_model_state': True,
}


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
            writer=None, use_cuda=False, snapshot_model_state=True):
    # target CPU or GPU
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print("Using device:")
    print(device)

    model.to(device)
    start_time = time.time()
    for i in range(epochs):
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
            model.total_batches += 1
            loss.backward()
            optimizer.step()
            if not writer is None:
                writer.add_scalar('loss', loss, model.total_batches)
            del x, loss, labels

            if model.total_batches % checkpoint_interval == 0:
                metrics = evaluate_model(model, valid_loader, device, disable_eval)
                metrics['avg_loss'] = loss_sum / checkpoint_interval
                elapsed_time = time.time() - start_time
                metrics['elapsed_time_sec'] = elapsed_time

                print("epoch {}, batch {}, {}".format(i, model.total_batches, metrics))

                if not writer is None:
                    for key, value in metrics.items():
                        writer.add_scalar(key, value, global_step=model.total_batches, walltime=elapsed_time)
                    if snapshot_model_state:
                        log_model_state(model, writer, model.total_batches, elapsed_time, include_gradients=True)
                yield model, i, model.total_batches, metrics
                loss_sum = 0

            optimizer.zero_grad()

    metrics = evaluate_model(model, valid_loader, device, disable_eval)
    if not writer is None:
        for key, value in metrics.items():
            writer.add_scalar(key, value, global_step=model.total_batches, walltime=elapsed_time)
        if snapshot_model_state:
            log_model_state(model, writer, model.total_batches, elapsed_time, include_gradients=False)
    yield model, epochs, len(train_loader), metrics


def log_model_state(model, writer, batch, elapsed_time, include_gradients=True, scale_factor=20):
    for key, value in model.named_parameters():
        tensor_shape = value.shape
        if len(tensor_shape) == 0:  # if 0D, output 1x1 image
            new_shape = (1, 1, 1, 1)
        elif len(tensor_shape) == 1:  # if 1D, width = in_dimension, height = 1
            new_shape = (1, 1, 1, tensor_shape[0])
        elif len(tensor_shape) == 2:  # if 2D, width = in_dimension, height = out_channels
            new_shape = (1, 1, tensor_shape[0], tensor_shape[1])
        elif len(tensor_shape) == 3:  # if 3D, width = in_dimension, height = in_channels, grid = out_channels
            new_shape = (tensor_shape[0], 1, tensor_shape[1], tensor_shape[2])
        
        img = value.cpu().detach().reshape(new_shape)
        writer.add_histogram('hist_' + key, img.flatten(), global_step=batch, walltime=elapsed_time)

        if include_gradients:
            grad = value.grad.cpu().detach().reshape(new_shape)
            writer.add_histogram('grad_' + key, grad.flatten(), global_step=batch, walltime=elapsed_time)
            pos_grad = F.relu(grad)
            neg_grad = F.relu(-1 * grad)
            img = torch.cat((img + pos_grad * scale_factor, img, img + neg_grad * scale_factor), dim=1)

        img = make_grid(img, pad_value=2)
        writer.add_image(key, img, global_step=batch, walltime=elapsed_time, dataformats='CHW')


def load_model(config):
    loss_fn = NeighbourDistanceLoss(config['neighbour_loss_prop'])

    if config['model'] == 'Multilayer' or config['model'] == 'LatentLinearRegression':
        model = MultilayerEncoder(config['kernel_len'], config['latent_len'], config['seq_len'],
                config['seq_per_batch'], config['input_dropout_freq'], config['latent_noise_std'], loss_fn,
                config['hidden_len'], config['pool_size'], config['n_conv_and_pool'],
                config['n_conv_before_pool'], config['n_linear'], config['hidden_dropout_freq'])
    else:
        model = Autoencoder(config['kernel_len'], config['latent_len'], config['seq_len'],
                config['seq_per_batch'], config['input_dropout_freq'], config['latent_noise_std'], loss_fn)

    if not (config['load_prev_model_state'] is None):
        state_dict = torch.load(config['load_prev_model_state'], map_location=torch.device('cpu'))
        if not '_total_batches' in state_dict:
            state_dict['_total_batches'] = torch.tensor([[0]], dtype=torch.long)
        model.load_state_dict(state_dict)

    if config['model'] == 'LatentLinearRegression':
        encoder = model
        model = LatentLinearRegression(config['kernel_len'], config['latent_len'], config['seq_len'],
                config['seq_per_batch'], config['input_dropout_freq'], config['latent_noise_std'], loss_fn,
                encoder, config['output_len'])

    return model


def load_dataset(config):
    if config['model'] == 'Autoencoder':
        dataset = SequenceDataset(config['input_path'], seq_len=config['seq_len'], overlap=(config['kernel_len'] - 1))
    elif config['model'] == 'LatentLinearRegression':
        dataset = LabelledSequence(config['input_path'], config['seq_len'])
    else:
        dataset = SequenceDataset(config['input_path'], config['seq_len'])
    return dataset


def get_dataloaders(config):
    dataset = load_dataset(config)
    split_size = int(config['split_prop'] * len(dataset))
    if split_size == len(dataset):
        split_size = len(dataset) - config['seq_per_batch']
    train_data, valid_data = torch.utils.data.random_split(dataset, [len(dataset) - split_size, split_size])
    train_loader = torch.utils.data.DataLoader(train_data,
            batch_size=config['seq_per_batch'], shuffle=True, num_workers=config['n_dataloader_workers'])
    valid_loader = torch.utils.data.DataLoader(valid_data,
            batch_size=config['seq_per_batch']*2, shuffle=False, num_workers=config['n_dataloader_workers'])
    return train_loader, valid_loader


def run(config):
    config = update_config(config)
    print("Config values:")
    print(config)
    
    sub_dir = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
    writer = SummaryWriter(output_path(config['log_path'], directory=sub_dir))

    if config['fixed_random_seed']:
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    print("Loading model...")
    model = load_model(config)
    print(model)

    print("Loading data...")
    train_loader, valid_loader = get_dataloaders(config)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learn_rate'])

    print("Training for {} epochs".format(config['epochs']))
    checkpoints = train(model, train_loader, valid_loader, optimizer, config['epochs'],
            config['disable_eval'], config['checkpoint_interval'], writer,
            use_cuda=config['use_cuda_if_available'], snapshot_model_state=config['snapshot_model_state'])

    for model, i, j, metrics in checkpoints:
        print("Model evaluation at epoch {}, batch {}, metrics {}".format(i, j, metrics))
        if config['save_model']:
            model_str = "{}{}x{}d{}n{}l{}_{}at{}_{}-{}".format(config['model'],
                    model.kernel_len, model.latent_len, model.input_dropout_freq,
                    model.latent_noise_std, config['neighbour_loss_prop'],
                    model.total_batches + config['epochs'], config['learn_rate'], i, j)
            out_file = output_path(config['name'], config['input_path'], model_str + '.pth', directory=sub_dir)
            print("Saving model to {}".format(out_file))
            torch.save(model.state_dict(), out_file)

    return model, train_loader, valid_loader, optimizer, loss_fn


def update_config(config):
    defaults = dict(__CONFIG_DEFAULT)
    defaults.update(config)
    return defaults


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Configuration and hyperparameters for autoencoder.')
    for key, value in __CONFIG_DEFAULT.items():
        parser.add_argument('--' + key, type=type(value), required=False, default=value)
    config = parser.parse_args()
    run(vars(config))
