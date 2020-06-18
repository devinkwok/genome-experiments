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
        'kernel_len': 1,
        'latent_len': 1,
        'seq_len': 32,
        'seq_per_batch': 4,
        'input_path': "data/ref_genome/test.fasta",
        'split_prop': 0.2,
        'epochs': 20,
        'learn_rate': 0.01,
        'input_dropout_freq': 0.0,
        'latent_noise_std': 0.0,
    }
    # one latent var isn't enough, >= 2 bits should capture all 4 bp
    experiment(hparams, 'latent_len', [1, 2, 4])

    # similar experiment on window size 3, >= 2 * 3 bits should capture all variation
    hparams['kernel_len'] = 3
    experiment(hparams, 'latent_len', [2, 4, 6, 12])


def exp2_input_drop():
    hparams = {
        'name': "ae01-drop",
        'kernel_len': 1,
        'latent_len': 1,
        'seq_len': 32,
        'seq_per_batch': 4,
        'input_path': "data/ref_genome/test.fasta",
        'split_prop': 0.2,
        'epochs': 20,
        'learn_rate': 0.01,
        'input_dropout_freq': 0.2,
        'latent_noise_std': 0.0,
    }
    # dropout should reduce accuracy by similar amount
    experiment(hparams, 'latent_len', [1, 2, 4])
    experiment(hparams, 'latent_len', [2, 4, 6, 12])


def exp3_latent_noise():
    # adding more noise to latent variable on pre-trained network should reduce performance
    hparams = {
        'name': "ae01-noise",
        'kernel_len': 3,
        'latent_len': 6,
        'seq_len': 32,
        'seq_per_batch': 4,
        'input_path': "data/ref_genome/test.fasta",
        'split_prop': 0.9,  # disable training
        'epochs': 5,  # 5 runs of validation
        'learn_rate': 0.0,  # disable training
        'input_dropout_freq': 0.0,
        'load_prev_model_state': "outputs/src/ae/autoencoder/ae01droptest3x6_drop0.2_20_at0.01.pth",
        'save_model': False,
        'disable_eval': True,
    }
    experiment(hparams, 'latent_noise_std', [0.0, 0.2, 0.5, 1.0, 2.0, 5.0])


def exp4_larger_windows():
    hparams = {
        'name': "ae01",
        'kernel_len': 6,
        'latent_len': 1,
        'seq_len': 32,
        'seq_per_batch': 4,
        'input_path': "data/ref_genome/test.fasta",
        'split_prop': 0.2,
        'epochs': 20,
        'learn_rate': 0.01,
        'input_dropout_freq': 0.1,
        'latent_noise_std': 0.3,
    }
    experiment(hparams, 'latent_len', [6, 9, 12, 18, 24])
    hparams['kernel_len'] = 12
    experiment(hparams, 'latent_len', [12, 18, 24, 30])


def exp5_neighbourhood_loss():
    hparams = {
        'name': "ae01-nbl",
        'kernel_len': 8,
        'latent_len': 16,
        'seq_len': 32,
        'seq_per_batch': 4,
        'input_path': "data/ref_genome/test.fasta",
        'split_prop': 0.2,
        'epochs': 20,
        'learn_rate': 0.01,
        'input_dropout_freq': 0.1,
        'latent_noise_std': 0.3,
    }
    experiment(hparams, 'neighbour_loss_prop', [0, 0.1, 0.2, 0.5, 1.0])


def exp6_multilayer():
    hparams = {
        'model': 'Multilayer',
        'name': "aem0",
        'kernel_len': 3,
        'latent_len': 30,
        'seq_len': 27,
        'seq_per_batch': 20,
        'input_path': "data/ref_genome/chr22_excerpt_800k.fa",
        'split_prop': 0.05,
        'epochs': 5,
        'learn_rate': 0.01,
        'input_dropout_freq': 0.05,
        'latent_noise_std': 0.3,
        'hidden_len': 10,
        'pool_size': 3,
        'n_conv_and_pool': 2,
        'n_conv_before_pool': 1,
        'n_linear': 1,
        'neighbour_loss_prop': 0.0,
    }
    experiment(hparams, 'learn_rate', [10., 5., 2.])
    # experiment(hparams, 'n_conv_before_pool', [1, 2, 3])


def exp7_multilayer_long():
    hparams = {
        'model': 'Multilayer',
        'name': "aem0",
        'kernel_len': 9,
        'latent_len': 240,
        'seq_len': 64,
        'seq_per_batch': 20,
        'input_path': "data/ref_genome/chr22_excerpt_4m.fa",
        # 'load_prev_model_state': "outputs/src/ae/autoencoder/aem0chr22_excerpt_4mMultilayer3x30d0.05n0.3l0.0_20at2.0.pth",
        'split_prop': 0.05,
        'epochs': 5,
        'learn_rate': 1.,
        'input_dropout_freq': 0.05,
        'latent_noise_std': 0.3,
        'hidden_len': 36,
        'pool_size': 4,
        'n_conv_and_pool': 2,
        'n_conv_before_pool': 1,
        'n_linear': 1,
        'neighbour_loss_prop': 0.0,
    }
    experiment(hparams, 'latent_len', [80, 160, 240])
    experiment(hparams, 'hidden_len', [12, 24, 36])


def exp8_multilayer_rerun():
    hparams = {
        'model': 'Multilayer',
        'name': "aem0",
        'kernel_len': 9,
        'latent_len': 200,
        'seq_len': 64,
        'seq_per_batch': 40,
        'input_path': "data/ref_genome/chr22.fa",
        # 'load_prev_model_state': "outputs/src/ae/autoencoder/aem0chr22_excerpt_4mMultilayer3x30d0.05n0.3l0.0_20at2.0.pth",
        'split_prop': 0.05,
        'epochs': 20,
        'learn_rate': 0.1,
        'input_dropout_freq': 0.03,
        'latent_noise_std': 0.2,
        'hidden_len': 24,
        'pool_size': 4,
        'n_conv_and_pool': 2,
        'n_conv_before_pool': 2,
        'n_linear': 2,
        'neighbour_loss_prop': 0.0,
    }
    experiment(hparams, 'input_dropout_freq', [0.0, 0.02, 0.05, 0.0, 0.02, 0.05, 0.0, 0.02, 0.05])

if __name__ == '__main__':
    # for reproducibility
    DEBUG = True
    if DEBUG:
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # exp1_window_size()
    # exp2_input_drop()
    # exp3_latent_noise()
    # exp4_larger_windows()
    # exp5_neighbourhood_loss()
    # exp6_multilayer()
    # exp7_multilayer_long()
    exp8_multilayer_rerun()
