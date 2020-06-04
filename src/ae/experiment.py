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


def experiment_1(hparams):
    # one latent var isn't enough, >= 2 bits should capture all 4 bp
    ae.run(hparams)

    hparams['latent_len'] = 2
    ae.run(hparams)

    hparams['latent_len'] = 4
    ae.run(hparams)

    # similar experiment on window size 3, >= 2 * 3 bits should capture all variation
    hparams['window_len'] = 3
    hparams['latent_len'] = 2
    ae.run(hparams)

    hparams['latent_len'] = 4
    ae.run(hparams)

    hparams['latent_len'] = 6
    ae.run(hparams)

    hparams['latent_len'] = 12
    ae.run(hparams)

    hparams['latent_len'] = 24
    ae.run(hparams)


# adding more noise to latent variable on pre-trained network should reduce performance
def experiment_2(hparams):
    # no noise is baseline performance
    hparams['noise_std'] = 0.0
    ae.run(hparams)

    # performance should degrade with increased noise
    hparams['noise_std'] = 0.2
    ae.run(hparams)

    hparams['noise_std'] = 0.5
    ae.run(hparams)

    hparams['noise_std'] = 1
    ae.run(hparams)

    hparams['noise_std'] = 2
    ae.run(hparams)

    hparams['noise_std'] = 5
    ae.run(hparams)


if __name__ == '__main__':
    # for reproducibility
    DEBUG = True
    if DEBUG:
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
    # experiment_1(hparams)

    hparams['name'] = "ae01drop"
    hparams['dropout_freq'] = 0.2
    # experiment_1(hparams)

    hparams = {
        'name': "ae01noise",
        'window_len': 3,
        'latent_len': 6,
        'seq_len': 32,
        'seq_per_batch': 4,
        'input_path': "data/ref_genome/test.fasta",
        'split_prop': 0.9,  # disable training
        'epochs': 5,  # 5 runs of validation
        'learn_rate': 0.0,  # disable training
        'dropout_freq': 0.0,
        'prev_model_state': "outputs/src/ae/autoencoder/ae01droptest3x6_drop0.2_20_at0.01.pth",
        'save_model': False,
        'set_eval': False,
    }
    experiment_2(hparams)
