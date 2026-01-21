#!/bin/bash
# Download CLIP-MIA-Bench dataset and models from Hugging Face
# Usage: ./download_assets.sh

set -e  # Exit on any error

#-----------------------------
# Configuration
#-----------------------------
HF_DATASET_REPO="https://huggingface.co/datasets/CLIP-MIA-Bench/clip-mia-bench-data"
HF_MODEL_REPO="https://huggingface.co/CLIP-MIA-Bench/clip-mia-bench-models"


#-----------------------------
# Helper function
#-----------------------------
function download_repo() {
    local REPO_URL=$1

    echo "Cloning $REPO_URL into $TARGET_DIR ..."
    git lfs install
    git clone "$REPO_URL" 
}

#-----------------------------
# Download dataset
#-----------------------------
echo "==> Downloading dataset..."
download_repo $HF_DATASET_REPO 

#-----------------------------
# Download models
#-----------------------------
echo "==> Downloading models..."
download_repo $HF_MODEL_REPO 

echo "==> All assets downloaded successfully!"

