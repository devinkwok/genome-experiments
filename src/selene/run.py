from selene_sdk.utils import load_path
from selene_sdk.utils import parse_configs_and_run

import sys
sys.path.append('./src/')
import glob
from os.path import join


def run_configs(directory, lr):
    for file in glob.glob(join(directory, '*.yml')):
        print(file)
        parse_configs_and_run(load_path(file), lr=lr)


if __name__ == '__main__':
    lr = 0.1
    directory = './src/selene/configs'
    run_configs(directory, lr)


    # # parse_configs_and_run(load_path("./src/selene/configs/config_deepsea1.yml"), lr=lr)
    # # parse_configs_and_run(load_path("./src/selene/configs/config_deepsea2.yml"), lr=lr)
    # # parse_configs_and_run(load_path("./src/selene/configs/config_deepsea4.yml"), lr=lr)
    # # parse_configs_and_run(load_path("./src/selene/configs/config_deepsea8.yml"), lr=lr)
    # # parse_configs_and_run(load_path("./src/selene/configs/config_deepsea16.yml"), lr=lr)
    # # # parse_configs_and_run(load_path("./src/selene/configs/config_reversecomplementTT1.yml"), lr=lr)
    # # # parse_configs_and_run(load_path("./src/selene/configs/config_reversecomplementTT2.yml"), lr=lr)
    # # # parse_configs_and_run(load_path("./src/selene/configs/config_reversecomplementTT4.yml"), lr=lr)
    # # # parse_configs_and_run(load_path("./src/selene/configs/config_reversecomplementTT8.yml"), lr=lr)
    # # # parse_configs_and_run(load_path("./src/selene/configs/config_reversecomplementTT16.yml"), lr=lr)
    # # parse_configs_and_run(load_path("./src/selene/configs/config_nopoolTT1.yml"), lr=lr)
    # # parse_configs_and_run(load_path("./src/selene/configs/config_nopoolTT2.yml"), lr=lr)
    # # parse_configs_and_run(load_path("./src/selene/configs/config_nopoolTT4.yml"), lr=lr)
    # # parse_configs_and_run(load_path("./src/selene/configs/config_nopoolTT8.yml"), lr=lr)
    # # parse_configs_and_run(load_path("./src/selene/configs/config_nopoolTT16.yml"), lr=lr)
    # parse_configs_and_run(load_path("./src/selene/configs/config_reversecomplementTTTFTF1.yml"), lr=lr)
    # parse_configs_and_run(load_path("./src/selene/configs/config_reversecomplementTTTFTF2.yml"), lr=lr)
    # parse_configs_and_run(load_path("./src/selene/configs/config_reversecomplementTTTFTF4.yml"), lr=lr)
    # parse_configs_and_run(load_path("./src/selene/configs/config_reversecomplementTTTFTF8.yml"), lr=lr)
    # parse_configs_and_run(load_path("./src/selene/configs/config_reversecomplementTTTFTF16.yml"), lr=lr)
    # parse_configs_and_run(load_path("./src/selene/configs/config_nopoolTTTFTF1.yml"), lr=lr)
    # parse_configs_and_run(load_path("./src/selene/configs/config_nopoolTTTFTF2.yml"), lr=lr)
    # parse_configs_and_run(load_path("./src/selene/configs/config_nopoolTTTFTF4.yml"), lr=lr)
    # parse_configs_and_run(load_path("./src/selene/configs/config_nopoolTTTFTF8.yml"), lr=lr)
    # parse_configs_and_run(load_path("./src/selene/configs/config_nopoolTTTFTF16.yml"), lr=lr)
    # parse_configs_and_run(load_path("./src/selene/configs/config_reversecomplementTTTFTF1_half.yml"), lr=lr)
    # parse_configs_and_run(load_path("./src/selene/configs/config_reversecomplementTTTFTF2_half.yml"), lr=lr)
    # parse_configs_and_run(load_path("./src/selene/configs/config_reversecomplementTTTFTF4_half.yml"), lr=lr)
    # parse_configs_and_run(load_path("./src/selene/configs/config_reversecomplementTTTFTF8_half.yml"), lr=lr)
    # parse_configs_and_run(load_path("./src/selene/configs/config_reversecomplementTTTFTF16_half.yml"), lr=lr)
    # parse_configs_and_run(load_path("./src/selene/configs/config_nopoolTTTFTF1_half.yml"), lr=lr)
    # parse_configs_and_run(load_path("./src/selene/configs/config_nopoolTTTFTF2_half.yml"), lr=lr)
    # parse_configs_and_run(load_path("./src/selene/configs/config_nopoolTTTFTF4_half.yml"), lr=lr)
    # parse_configs_and_run(load_path("./src/selene/configs/config_nopoolTTTFTF8_half.yml"), lr=lr)
    # parse_configs_and_run(load_path("./src/selene/configs/config_nopoolTTTFTF16_half.yml"), lr=lr)
    # # parse_configs_and_run(load_path("./src/selene/configs/config_copykernel1.yml"), lr=lr)
    # # parse_configs_and_run(load_path("./src/selene/configs/config_copykernel2.yml"), lr=lr)
    # # parse_configs_and_run(load_path("./src/selene/configs/config_copykernel3.yml"), lr=lr)
    # # parse_configs_and_run(load_path("./src/selene/configs/config_retrainkernel1.yml"), lr=lr)
    # # parse_configs_and_run(load_path("./src/selene/configs/config_retrainkernel2.yml"), lr=lr)
    # # parse_configs_and_run(load_path("./src/selene/configs/config_retrainkernel3.yml"), lr=lr)
