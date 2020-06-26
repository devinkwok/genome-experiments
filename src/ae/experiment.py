import timeit

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


def experiment(hparams, key, values, n_runs=1):
    for value in values:
        hparams[key] = value
        run_fn = lambda: ae.run(hparams)
        runtime = timeit.timeit(run_fn, number=n_runs)
        print('Runtime: ', runtime)


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
        # 'input_path': "data/ref_genome/chr22.fa",
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
        'hidden_dropout_freq': 0.05,
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
        'hidden_dropout_freq': 0.05,
    }
    experiment(hparams, 'latent_len', [80, 160, 240])
    experiment(hparams, 'hidden_len', [12, 24, 36])


def exp8_multilayer_rerun():
    hparams = {
        'model': 'Multilayer',
        'name': "aem0h",
        'kernel_len': 9,
        'latent_len': 200,
        'seq_len': 64,
        'seq_per_batch': 200,
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
        'hidden_dropout_freq': 0.05,
        'n_dataloader_workers': 4,
        'checkpoint_interval': 1000000,
    }
    experiment(hparams, 'learn_rate', [0.1, 0.01, 0.001])

def exp9_latentlinear():

    hparams = {
        'name': 'linear',
        'model': 'LatentLinearRegression',
        'kernel_len': 9,
        'latent_len': 200,
        'seq_len': 64,
        'seq_per_batch': 200,
        'input_path': 'data/deepsea/deepsea-test.pyc',
        'split_prop': 0.05,
        'epochs': 2,
        'learn_rate': 0.0001,
        'input_dropout_freq': 0.0,
        'latent_noise_std': 0.0,
        'save_model': True,
        'disable_eval': False,
        'neighbour_loss_prop': 0.0,
        # 'load_prev_model_state': 'outputs/src/ae/experiment/aem0hchr22Multilayer9x200d0.03n0.2l0.0_3at1.0_0-79999.pth',
        'load_prev_model_state': 'outputs/src/ae/experiment/aem0hchr22Multilayer9x200d0.03n0.2l0.0_4at0.01_1-153990.pth',
        'hidden_len': 24,
        'pool_size': 4,
        'n_conv_and_pool': 2,
        'n_conv_before_pool': 2,
        'n_linear': 2,
        'use_cuda_if_available': True,
        'hidden_dropout_freq': 0.0,
        'fixed_random_seed': True,
        'n_dataloader_workers': 4,
        'checkpoint_interval': 1,
        'output_len': 2,      # for supervised model
    }
    experiment(hparams, 'learn_rate', [0.01, 0.001])


def exp10_TEST_use_old_dataset():

    hparams = {
        'name': 'TEST_use_old_dataset',
        'model': 'Autoencoder',
        'kernel_len': 3,
        'latent_len': 6,
        'seq_len': 40,
        'seq_per_batch': 20,
        'input_path': 'data/ref_genome/chr22_excerpt_800k.fa',
        'split_prop': 0.05,
        'epochs': 2,
        'learn_rate': 0.1,
        'input_dropout_freq': 0.0,
        'latent_noise_std': 0.2,
        'save_model': True,
        'disable_eval': True,
        'neighbour_loss_prop': 0.0,
        'use_cuda_if_available': True,
        'fixed_random_seed': True,
        'n_dataloader_workers': 4,
        'checkpoint_interval': 500,
        'TEST_use_old_dataset': False,
        'TEST_get_as_onehot': True,
        'TEST_get_label': False,
    }
    experiment(hparams, 'TEST_use_old_dataset', [False, True], n_runs=2)


def exp11_TEST_get_as_onehot():

    hparams = {
        'name': 'TEST_get_as_onehot',
        'model': 'Autoencoder',
        'kernel_len': 3,
        'latent_len': 6,
        'seq_len': 40,
        'seq_per_batch': 20,
        'input_path': 'data/ref_genome/chr22_excerpt_800k.fa',
        'split_prop': 0.05,
        'epochs': 2,
        'learn_rate': 0.1,
        'input_dropout_freq': 0.0,
        'latent_noise_std': 0.2,
        'save_model': True,
        'disable_eval': True,
        'neighbour_loss_prop': 0.0,
        'use_cuda_if_available': True,
        'fixed_random_seed': True,
        'n_dataloader_workers': 4,
        'checkpoint_interval': 100,
        'TEST_use_old_dataset': False,
        'TEST_get_as_onehot': False,
        'TEST_get_label': False,
    }
    experiment(hparams, 'TEST_get_as_onehot', [False, True], n_runs=2)

    hparams = {
        'name': 'TEST_get_as_onehot_multilayer',
        'model': 'MultilayerAutoencoder',
        'kernel_len': 9,
        'latent_len': 200,
        'seq_len': 64,
        'seq_per_batch': 100,
        'input_path': 'data/ref_genome/chr22_excerpt_800k.fa',
        'split_prop': 0.05,
        'epochs': 1,
        'learn_rate': 0.1,
        'input_dropout_freq': 0.05,
        'latent_noise_std': 0.2,
        'hidden_len': 24,
        'pool_size': 4,
        'n_conv_and_pool': 2,
        'n_conv_before_pool': 2,
        'n_linear': 2,
        'hidden_dropout_freq': 0.05,
        'save_model': True,
        'disable_eval': True,
        'neighbour_loss_prop': 0.0,
        'use_cuda_if_available': True,
        'fixed_random_seed': True,
        'n_dataloader_workers': 4,
        'checkpoint_interval': 10000,
        'TEST_use_old_dataset': False,
        'TEST_get_as_onehot': False,
        'TEST_get_label': False,
    }
    experiment(hparams, 'TEST_get_as_onehot', [False, True], n_runs=2)


def exp12_TEST_get_label():

    hparams = {
        'name': 'TEST_get_label',
        'model': 'Autoencoder',
        'kernel_len': 3,
        'latent_len': 6,
        'seq_len': 40,
        'seq_per_batch': 20,
        'input_path': 'data/ref_genome/chr22_excerpt_800k.fa',
        'split_prop': 0.05,
        'epochs': 2,
        'learn_rate': 0.1,
        'input_dropout_freq': 0.0,
        'latent_noise_std': 0.2,
        'save_model': True,
        'disable_eval': True,
        'neighbour_loss_prop': 0.0,
        'use_cuda_if_available': True,
        'fixed_random_seed': True,
        'n_dataloader_workers': 4,
        'checkpoint_interval': 100,
        'TEST_use_old_dataset': False,
        'TEST_get_as_onehot': False,
        'TEST_get_label': False,
    }
    experiment(hparams, 'TEST_get_label', [False, True], n_runs=2)

    hparams = {
        'name': 'TEST_get_label_multilayer',
        'model': 'MultilayerAutoencoder',
        'kernel_len': 9,
        'latent_len': 200,
        'seq_len': 64,
        'seq_per_batch': 100,
        'input_path': 'data/ref_genome/chr22_excerpt_800k.fa',
        'split_prop': 0.05,
        'epochs': 1,
        'learn_rate': 0.1,
        'input_dropout_freq': 0.05,
        'latent_noise_std': 0.2,
        'hidden_len': 24,
        'pool_size': 4,
        'n_conv_and_pool': 2,
        'n_conv_before_pool': 2,
        'n_linear': 2,
        'hidden_dropout_freq': 0.05,
        'save_model': True,
        'disable_eval': True,
        'neighbour_loss_prop': 0.0,
        'use_cuda_if_available': True,
        'fixed_random_seed': True,
        'n_dataloader_workers': 4,
        'checkpoint_interval': 10000,
        'TEST_use_old_dataset': False,
        'TEST_get_as_onehot': False,
        'TEST_get_label': False,
    }
    experiment(hparams, 'TEST_get_label', [False, True], n_runs=2)

def exp13_test_flags():

    hparams = {
        'name': 'TEST_get_label',
        'model': 'Autoencoder',
        'kernel_len': 3,
        'latent_len': 6,
        'seq_len': 40,
        'seq_per_batch': 20,
        'input_path': 'data/ref_genome/test.fasta',
        'split_prop': 0.05,
        'epochs': 2,
        'learn_rate': 0.1,
        'input_dropout_freq': 0.0,
        'latent_noise_std': 0.2,
        'save_model': True,
        'disable_eval': True,
        'neighbour_loss_prop': 0.0,
        'use_cuda_if_available': True,
        'fixed_random_seed': True,
        'n_dataloader_workers': 4,
        'checkpoint_interval': 100,
        'TEST_use_old_dataset': False,
        'TEST_get_as_onehot': True,
        'TEST_get_label': False,
    }
    experiment(hparams, 'TEST_use_old_dataset', [True], n_runs=1)

    hparams = {
        'name': 'TEST_get_label_multilayer',
        'model': 'MultilayerAutoencoder',
        'kernel_len': 9,
        'latent_len': 200,
        'seq_len': 64,
        'seq_per_batch': 100,
        'input_path': 'data/ref_genome/test.fasta',
        'split_prop': 0.05,
        'epochs': 1,
        'learn_rate': 0.1,
        'input_dropout_freq': 0.05,
        'latent_noise_std': 0.2,
        'hidden_len': 24,
        'pool_size': 4,
        'n_conv_and_pool': 2,
        'n_conv_before_pool': 2,
        'n_linear': 2,
        'hidden_dropout_freq': 0.05,
        'save_model': True,
        'disable_eval': True,
        'neighbour_loss_prop': 0.0,
        'use_cuda_if_available': True,
        'fixed_random_seed': True,
        'n_dataloader_workers': 4,
        'checkpoint_interval': 10000,
        'TEST_use_old_dataset': True,
        'TEST_get_as_onehot': True,
        'TEST_get_label': False,
    }
    experiment(hparams, 'TEST_use_old_dataset', [True], n_runs=1)



if __name__ == '__main__':
    # exp1_window_size()
    # exp2_input_drop()
    # exp3_latent_noise()
    # exp4_larger_windows()
    # exp5_neighbourhood_loss()
    # exp6_multilayer()
    # exp7_multilayer_long()
    # exp8_multilayer_rerun()
    # exp9_latentlinear()

    exp10_TEST_use_old_dataset()
    exp11_TEST_get_as_onehot()
    exp12_TEST_get_label()

    # exp13_test_flags()