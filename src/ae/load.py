import torch

from ae.autoencoder import NeighbourDistanceLoss, Autoencoder, MultilayerEncoder, LatentLinearRegression

def load_model(config):
    if config['loss_fn'] == 'cross_entropy_loss':
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
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
