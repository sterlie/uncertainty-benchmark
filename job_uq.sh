#!/bin/bash
#BSUB -q gpul40s
#BSUB -J uq_isic_drop
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=5GB]"
#BSUB -M 6GB
#BSUB -W 24:00
#BSUB -u sarste@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o logs/Output_%J.out
#BSUB -e logs/Error_%J.err

# Enable debugging (prints commands and stops on errors)
set -x
set -e

# Load Python module available on DTU HPC
module load python/3.11.7

# Create logs directory if it does not exist
mkdir -p logs

# Create virtual environment if it doesn't exist
if [ ! -d "uq" ]; then
    python3 -m venv uq
fi

# Activate virtual environment
source uq/bin/activate


# Upgrade pip and install dependencies (no cache to save space)
python -m pip install --upgrade pip --no-cache-dir
python -m pip install --no-cache-dir -r requirements.txt


make run-isic  EXPERIMENT=isic_drop      MODEL=EfficientNet       METHOD=all_methods


