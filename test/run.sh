#!/bin/bash
#SBATCH --job-name=cupy_test     # Job name
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --cpus-per-task=4        # CPU cores per task
#SBATCH --mem=16G                # Memory per node
#SBATCH --time=00:10:00          # Time limit hrs:min:sec
#SBATCH --output=output_%j.log   # Standard output and error log

# Load necessary modules and activate environment
module load cuda/12.6
source ../bien410/bin/activate

# Run the Python script
python gpu.py
