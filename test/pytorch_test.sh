#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=00:20:00
#SBATCH --mail-user=yitao.sun@mail.mcgill.ca
#SBATCH --mail-type=ALL

module load python # Make sure to choose a version that suits your application
# virtualenv --no-download $SLURM_TMPDIR/env
# source $SLURM_TMPDIR/env/bin/activate
# pip install torch --no-index

cd ~/$project/
source ~/bien410/bien410/bien410/bin/activate
cd ~/bien410/bien410/test/


python pytorch_test.py