from selene_sdk.utils import load_path
from selene_sdk.utils import parse_configs_and_run

import sys
sys.path.append('./src/')

lr = 0.1
parse_configs_and_run(load_path("./src/selene/config_deepsea.yml"), lr=lr)
parse_configs_and_run(load_path("./src/selene/config_reversecomplement3.yml"), lr=lr)
parse_configs_and_run(load_path("./src/selene/config_reversecomplement5.yml"), lr=lr)
# parse_configs_and_run(load_path("./src/selene/config_copykernel.yml"), lr=lr)
