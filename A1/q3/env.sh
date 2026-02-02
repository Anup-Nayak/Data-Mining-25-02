#!/bin/bash

set -e

echo "Updating package list..."
sudo apt update

echo "Installing Python3 and required packages..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-numpy \
    python3-networkx \
    python3-joblib

echo "Setup complete."
