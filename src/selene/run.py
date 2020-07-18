from selene_sdk.utils import load_path
from selene_sdk.utils import parse_configs_and_run

import sys
sys.path.append('./src/')

lr = 0.1
# parse_configs_and_run(load_path("./src/selene/config_deepsea1.yml"), lr=lr)
# parse_configs_and_run(load_path("./src/selene/config_deepsea2.yml"), lr=lr)
# parse_configs_and_run(load_path("./src/selene/config_deepsea4.yml"), lr=lr)
# parse_configs_and_run(load_path("./src/selene/config_deepsea8.yml"), lr=lr)
# parse_configs_and_run(load_path("./src/selene/config_deepsea16.yml"), lr=lr)
# parse_configs_and_run(load_path("./src/selene/config_reversecomplementTT1.yml"), lr=lr)
# parse_configs_and_run(load_path("./src/selene/config_reversecomplementTT2.yml"), lr=lr)
# parse_configs_and_run(load_path("./src/selene/config_reversecomplementTT4.yml"), lr=lr)
# parse_configs_and_run(load_path("./src/selene/config_reversecomplementTT8.yml"), lr=lr)
# parse_configs_and_run(load_path("./src/selene/config_reversecomplementTT16.yml"), lr=lr)
# parse_configs_and_run(load_path("./src/selene/config_reversecomplementTTTFTF1.yml"), lr=lr)
# parse_configs_and_run(load_path("./src/selene/config_reversecomplementTTTFTF2.yml"), lr=lr)
# parse_configs_and_run(load_path("./src/selene/config_reversecomplementTTTFTF4.yml"), lr=lr)
# parse_configs_and_run(load_path("./src/selene/config_reversecomplementTTTFTF8.yml"), lr=lr)
# parse_configs_and_run(load_path("./src/selene/config_reversecomplementTTTFTF16.yml"), lr=lr)
# parse_configs_and_run(load_path("./src/selene/config_reversecomplementTTTFTF1_half.yml"), lr=lr)
# parse_configs_and_run(load_path("./src/selene/config_reversecomplementTTTFTF2_half.yml"), lr=lr)
# parse_configs_and_run(load_path("./src/selene/config_reversecomplementTTTFTF4_half.yml"), lr=lr)
# parse_configs_and_run(load_path("./src/selene/config_reversecomplementTTTFTF8_half.yml"), lr=lr)
# parse_configs_and_run(load_path("./src/selene/config_reversecomplementTTTFTF16_half.yml"), lr=lr)
parse_configs_and_run(load_path("./src/selene/config_retrainkernel1.yml"), lr=lr)
parse_configs_and_run(load_path("./src/selene/config_copykernel1.yml"), lr=lr)
parse_configs_and_run(load_path("./src/selene/config_copykernel2.yml"), lr=lr)
parse_configs_and_run(load_path("./src/selene/config_copykernel3.yml"), lr=lr)
parse_configs_and_run(load_path("./src/selene/config_retrainkernel2.yml"), lr=lr)
parse_configs_and_run(load_path("./src/selene/config_retrainkernel3.yml"), lr=lr)
