#!/bin/bash
#SBATCH --time=5-00:00:00
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --constraint=v100
#SBATCH -n 1
#SBATCH -o train_mats_%j.out
#SBATCH -e train_mats_%j.err

FILES=src/selene/configs/*.yml
for f in $FILES
do
    echo $f
    python -u src/selene/selene_cli.py $f --lr=0.08
done
