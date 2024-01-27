#!/bin/bash

# Update and Upgrade the System
echo "Updating and upgrading the system..."
sudo apt-get update -y
sudo apt-get upgrade -y

# Install dependencies for Gymnasium
echo "Installing dependencies for Gymnasium..."

# Development tools
sudo apt-get install -y build-essential 

# Python 3 and pip
sudo apt-get install -y python3 python3-pip

# System libraries
sudo apt-get install -y libglew-dev libjpeg-dev libboost-all-dev libglu1-mesa-dev freeglut3-dev mesa-common-dev

# SWIG for interface generation
sudo apt-get install -y swig

# Gymnasium and additional dependencies via pip
echo "Installing requirements.txt"
pip3 install requirements.txt

echo "Setup complete!"