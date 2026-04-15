#!/bin/bash
# setup.sh — Install all ChatBox client dependencies (non-Docker)
# Run once on the Jetson or Ubuntu machine:
#   chmod +x setup.sh && ./setup.sh

set -e

echo "=================================================="
echo "  ChatBox Client Setup"
echo "=================================================="

# System packages
echo ""
echo "Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y \
    ffmpeg \
    espeak espeak-data \
    alsa-utils \
    portaudio19-dev libportaudio2 libportaudiocpp0 \
    python3-pip python3-dev

# Python packages
echo ""
echo "Installing Python packages..."
pip3 install --upgrade pip
pip3 install websockets requests pyserial gtts numpy

# PyAudio (build from source to avoid _portaudio errors on Jetson)
echo ""
echo "Installing PyAudio..."
pip3 install --no-binary :all: pyaudio || pip3 install pyaudio

# Optional: RealSense (uncomment if using emotion module)
# pip3 install pyrealsense2

echo ""
echo "=================================================="
echo "  Setup complete!"
echo "  Next steps:"
echo "  1. Edit client_config.json — set ip_address and server_url"
echo "  2. python3 robot.py"
echo "=================================================="