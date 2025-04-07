#!/bin/bash

source /root/miniconda3/bin/activate
conda activate dev
echo "Using Conda env: $CONDA_DEFAULT_ENV"

python -u app.py
