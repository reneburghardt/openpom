#!/bin/bash
#SBATCH --job-name=training_pom
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rene.burghardt@uni-goettingen.de
#SBATCH --output=/user/burghardt17/u17926/.project/dir.project/rene/projects/openpom/slurm/slurm-%j.out
#SBATCH -t 48:00:00
#SBATCH -p react
#SBATCH --mem 64G
#SBATCH --cpus-per-gpu=8
#SBATCH -G 1

. ~/.bashrc

echo $CUDA_VISIBLE_DEVICES
nvidia-smi

micromamba activate odor-model

python /user/burghardt17/u17926/.project/dir.project/rene/projects/openpom/src/mpnn_pom_experanto_train.py