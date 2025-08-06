#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:04:00
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --account=lsms-ddcf
#SBATCH --mem=10G

module purge
module load gcc python cuda cudnn/8.9.7.29-12 openmpi py-torch

srun VENV/bin/python train_FNO.py