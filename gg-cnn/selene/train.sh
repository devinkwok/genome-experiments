#!/bin/bash
#SBATCH --time=5-00:00:00
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --constraint=v100
#SBATCH -n 1
#SBATCH -o train_mats_%j.out
#SBATCH -e train_mats_%j.err

source activate selene-env

python -u selene_cli.py ./configs/none_none.yml --lr=0.08
python -u selene_cli.py ./configs/rc_none.yml --lr=0.08
python -u selene_cli.py ./configs/all_none.yml --lr=0.08
python -u selene_cli.py ./configs/none_all_1.yml --lr=0.08
python -u selene_cli.py ./configs/rc_all_1.yml --lr=0.08
python -u selene_cli.py ./configs/all_all_1.yml --lr=0.08
python -u selene_cli.py ./configs/none_all_2.yml --lr=0.08
python -u selene_cli.py ./configs/rc_all_2.yml --lr=0.08
python -u selene_cli.py ./configs/all_all_2.yml --lr=0.08
