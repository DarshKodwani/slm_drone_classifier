#!/bin/bash

# Step 1: Delete existing virtual environment if it exists
VENV_DIR="qat-env"
if [ -d "$VENV_DIR" ]; then
    echo "Removing existing virtual environment..."
    rm -rf "$VENV_DIR"
fi

# Step 2: Create a new virtual environment
echo "Creating new virtual environment..."
python3 -m venv "$VENV_DIR"

# Step 3: Activate the virtual environment and install packages
echo "Activating virtual environment and installing dependencies..."
source "$VENV_DIR/bin/activate"
pip install -r requirements.txt

# Step 4: Source the virtual environment
echo "Virtual environment setup complete. You are now in the environment."
echo "To activate the environment again later, use: source $VENV_DIR/bin/activate"