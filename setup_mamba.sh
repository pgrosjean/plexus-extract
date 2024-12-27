#!/bin/bash
# This script uses CUDA 12.1. You can swap with CUDA 11.8.
mamba create --name plexus_extract \
    python=3.10 \
    pytorch-cuda=12.1 \
    pytorch=2.3.0 \
    cudatoolkit xformers -c pytorch -c nvidia -c xformers -c conda-forge \
    -y
mamba activate plexus_extract

pip install -e . -v