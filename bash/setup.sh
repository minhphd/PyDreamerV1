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
sudo apt-get install python3-opencv

# System libraries
sudo apt-get install -y libglew-dev libjpeg-dev libboost-all-dev libglu1-mesa-dev freeglut3-dev mesa-common-dev

# SWIG for interface generation
sudo apt-get install -y swig

# Gymnasium and additional dependencies via pip
echo "Installing requirements.txt"
pip3 install -r requirements.txt
sudo apt-get install xvfb
Xvfb :99 -screen 0 1024x768x24 &

export DISPLAY=:99

echo "Setup complete!"
