#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=20:00:00
#SBATCH --mail-user=yitao.sun@mail.mcgill.ca
#SBATCH --mail-type=ALL

module load python/3.12.4 # In Narval, I use python/3.12.4      all the packages installed using pip must be installed with python/3.12.4
python SStrain.py
