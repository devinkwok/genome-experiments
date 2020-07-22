from csv import writer
import numpy as np
from scipy.spatial.distance import cosine
import torch
import torch.nn.functional
from src.ae.train import *


def encode_sequence(config):
    config = update_config(config)
    model = load_model(config)
    dataset = SequenceDataset(config['input_path'], seq_len=config['seq_len'],
                overlap=(config['seq_len'] - 1), augment=False, no_gaps=False)
    data_loader = torch.utils.data.DataLoader(dataset,
            batch_size=config['seq_per_batch']* 2, shuffle=False,
            num_workers=config['n_dataloader_workers'])
    for batch in data_loader:
        yield model.encode(batch, TEST_new_onehot=True)


def write_latent_to_csv(config, csv_file):
    with open(csv_file, 'w', newline='') as file:
        csv_writer = writer(file)
        for latent_batch in encode_sequence(config):
            for row in latent_batch.detach().numpy():
                csv_writer.writerow(row)


def adjacent_cosine_similarity(csv_file):
    latent_vectors = np.genfromtxt(csv_file, delimiter=',')
    distance = []
    for a, b in zip(latent_vectors, latent_vectors[1:]):
        distance.append(cosine(a, b))
    return distance

if __name__ == '__main__':
    # config = {'name': 'aemd', 'model': 'Multilayer', 'kernel_len': 9,
    #             'latent_len': 200, 'seq_len': 64, 'seq_per_batch': 200,
    #             'input_path': 'data/ref_genome/test_2.fa', 'split_prop': 0.05, 'epochs': 1, 'learn_rate': 1.0, 'input_dropout_freq': 0.03, 'latent_noise_std': 0.2, 'save_model': True, 'disable_eval': False, 'neighbour_loss_prop': 0.0, 'load_prev_model_state': None, 'hidden_len': 24, 'pool_size': 4, 'n_conv_and_pool': 1, 'n_conv_before_pool': 2, 'n_linear': 2, 'use_cuda_if_available': True, 'hidden_dropout_freq': 0.05, 'fixed_random_seed': True, 'n_dataloader_workers': 4, 'checkpoint_interval': 10000, 'output_len': 919, 'log_path': 'logs', 'snapshot_model_state': True}
    # write_latent_to_csv(config, 'test_short.csv')
    np.savetxt('cosines.csv', adjacent_cosine_similarity('test_short.csv'))