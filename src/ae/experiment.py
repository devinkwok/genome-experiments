import torch

import autoencoder as ae


# this is needed if model changes architecture
# example to run below:
# rename_state_dict(hparams, {'decode.0.weight': 'decode.1.weight', 'decode.0.bias': 'decode.1.bias'})
def rename_state_dict(hparams, old_to_new_keys_dict):
    state_dict = torch.load(hparams['prev_model_state'])
    for old_key, new_key in old_to_new_keys_dict.items():
        if old_key in state_dict.keys():
            state_dict[new_key] = state_dict.pop(old_key)
    torch.save(state_dict, hparams['prev_model_state'])


def experiment(hparams, key, values):
    for value in values:
        hparams[key] = value
        ae.run(hparams)


def exp1_window_size():
    hparams = {
        'name': "ae01",
        'window_len': 1,
        'latent_len': 1,
        'seq_len': 32,
        'seq_per_batch': 4,
        'input_path': "data/ref_genome/test.fasta",
        'split_prop': 0.2,
        'epochs': 20,
        'learn_rate': 0.01,
        'dropout_freq': 0.0,
        'noise_std': 0.0,
    }
    # one latent var isn't enough, >= 2 bits should capture all 4 bp
    experiment(hparams, 'latent_len', [1, 2, 4])

    # similar experiment on window size 3, >= 2 * 3 bits should capture all variation
    hparams['window_len'] = 3
    experiment(hparams, 'latent_len', [2, 4, 6, 12])


def exp2_input_drop():
    hparams = {
        'name': "ae01-drop",
        'window_len': 1,
        'latent_len': 1,
        'seq_len': 32,
        'seq_per_batch': 4,
        'input_path': "data/ref_genome/test.fasta",
        'split_prop': 0.2,
        'epochs': 20,
        'learn_rate': 0.01,
        'dropout_freq': 0.2,
        'noise_std': 0.0,
    }
    # dropout should reduce accuracy by similar amount
    experiment(hparams, 'latent_len', [1, 2, 4])
    experiment(hparams, 'latent_len', [2, 4, 6, 12])


def exp3_latent_noise():
    # adding more noise to latent variable on pre-trained network should reduce performance
    hparams = {
        'name': "ae01-noise",
        'window_len': 3,
        'latent_len': 6,
        'seq_len': 32,
        'seq_per_batch': 4,
        'input_path': "data/ref_genome/test.fasta",
        'split_prop': 0.9,  # disable training
        'epochs': 5,  # 5 runs of validation
        'learn_rate': 0.0,  # disable training
        'dropout_freq': 0.0,
        'load_prev_model_state': "outputs/src/ae/autoencoder/ae01droptest3x6_drop0.2_20_at0.01.pth",
        'save_model': False,
        'disable_eval': True,
    }
    experiment(hparams, 'noise_std', [0.0, 0.2, 0.5, 1.0, 2.0, 5.0])


def exp4_larger_windows():
    hparams = {
        'name': "ae01",
        'window_len': 6,
        'latent_len': 1,
        'seq_len': 32,
        'seq_per_batch': 4,
        'input_path': "data/ref_genome/test.fasta",
        'split_prop': 0.2,
        'epochs': 20,
        'learn_rate': 0.01,
        'dropout_freq': 0.1,
        'noise_std': 0.3,
    }
    experiment(hparams, 'latent_len', [6, 9, 12, 18, 24])
    hparams['window_len'] = 12
    experiment(hparams, 'latent_len', [12, 18, 24, 30])


def exp5_neighbourhood_loss():
    hparams = {
        'name': "ae01-nbl",
        'window_len': 8,
        'latent_len': 16,
        'seq_len': 32,
        'seq_per_batch': 4,
        'input_path': "data/ref_genome/test.fasta",
        'split_prop': 0.2,
        'epochs': 20,
        'learn_rate': 0.01,
        'dropout_freq': 0.1,
        'noise_std': 0.3,
    }
    experiment(hparams, 'neighbour_loss_prop', [0, 0.1, 0.2, 0.5, 1.0])


if __name__ == '__main__':
    # for reproducibility
    DEBUG = True
    if DEBUG:
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    exp1_window_size()
    exp2_input_drop()
    exp3_latent_noise()
    exp4_larger_windows()
    exp5_neighbourhood_loss()
