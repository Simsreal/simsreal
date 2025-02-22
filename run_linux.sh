#!/bin/bash

if [ ! -f .env.example ]; then
    echo "Error: .env.example file not found"
    exit 1
fi

echo "Copying .env.linux.example to .env..."
cp .env.linux.example .env

if [ $? -eq 0 ]; then
    echo "Successfully copied .env.example to .env"
else
    echo "Error: Failed to copy .env.example to .env"
    exit 1
fi

if [ ! -f main.py ]; then
    echo "Error: main.py not found"
    exit 1
fi

echo "Running main.py..."
python main.py
