#!/bin/bash
set -e  # stop on first error

echo "=== Setting up environment ==="

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create model directory
mkdir -p models
cd models

echo "=== Downloading models from Hugging Face ==="
# Download models using huggingface-cli
huggingface-cli download GSAI-ML/LLaDA-8B-Instruct --local-dir LLaDA-8B-Instruct
huggingface-cli download Dream-org/Dream-v0-Instruct-7B --local-dir Dream-v0-Instruct-7B

cd ..

echo "=== Replacing Dream generation_utils.py ==="
TARGET_PATH="./models/Dream-v0-Instruct-7B/src/transformers/generation_utils.py"
CUSTOM_SRC="./src/model/dream_klass.py"

# Ensure target directory exists
if [ -f "$CUSTOM_SRC" ]; then
    if [ -f "$TARGET_PATH" ]; then
        echo "Backing up original generation_utils.py..."
        mv "$TARGET_PATH" "${TARGET_PATH}.bak"
    else
        echo "Warning: original generation_utils.py not found, creating new file..."
        mkdir -p "$(dirname "$TARGET_PATH")"
    fi

    echo "Copying custom generation_utils.py..."
    cp "$CUSTOM_SRC" "$TARGET_PATH"
else
    echo "Error: Custom source file $CUSTOM_SRC not found!"
    exit 1
fi

echo "Installation complete!"
echo "Models are in ./models/"
