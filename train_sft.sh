#!/bin/bash

#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 0-06:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH -p general
#SBATCH -q class
#SBATCH -A class_cse476spring2025
#SBATCH --mail-type=ALL
#SBATCH --mail-user="%u@asu.edu"
#SBATCH --export=NONE

module load mamba/latest
source activate cse476
cd ~/CSE476

accelerate launch train_sft_general.py
