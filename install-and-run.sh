#!/bin/bash

# Define the name of the virtual environment directory
VENV_DIR=".venv"

# Check if the virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    # Install python3.12-venv if on a system with apt-get (e.g., Debian/Ubuntu)
    if command -v apt-get &> /dev/null; then
        echo "Detected apt-get. Attempting to install python3.12-venv..."
        sudo apt-get update && sudo apt-get install -y python3.12-venv
    fi

    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists in $VENV_DIR."
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Skipping installation."
fi

# Run the application
echo "Starting the application..."
python3 -m mhn-localization
