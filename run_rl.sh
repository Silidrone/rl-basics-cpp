#!/bin/bash

# Check if cmake and nlohmann-json3-dev are installed
if ! command -v cmake &> /dev/null || ! dpkg -l | grep -q nlohmann-json3-dev; then
    echo "Installing dependencies (cmake and nlohmann-json3-dev)..."
    sudo apt update && sudo apt install cmake nlohmann-json3-dev -y
    echo "Dependencies installed successfully!"
fi

# Build and run the RL agent
cmake -S . -B build && cmake --build build && ./build/output_executable
