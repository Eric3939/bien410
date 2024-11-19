#!/bin/bash
#SBATCH --job-name=cupy_test     # Job name
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --cpus-per-task=4        # CPU cores per task
#SBATCH --mem=16G                # Memory per node
#SBATCH --time=00:01:00          # Time limit hrs:min:sec
#SBATCH --output=output_%j.log   # Standard output and error log

# Load necessary modules and activate environment
module load cuda
module load StdEnv/2023
module load arrow/14.0.1

source ../bien410/bin/activate

pip install cupy-cuda12x


# Run the Python script
python gpu.py
