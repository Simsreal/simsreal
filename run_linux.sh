#!/bin/bash

# 检查.env.example是否存在
if [ ! -f .env.example ]; then
    echo "Error: .env.example file not found"
    exit 1
fi

# 复制.env.example到.env
echo "Copying .env.example to .env..."
cp .env.example .env

# 检查复制是否成功
if [ $? -eq 0 ]; then
    echo "Successfully copied .env.example to .env"
else
    echo "Error: Failed to copy .env.example to .env"
    exit 1
fi

# 检查main.py是否存在
if [ ! -f main.py ]; then
    echo "Error: main.py not found"
    exit 1
fi

# 执行main.py（使用python命令）
echo "Running main.py..."
python3 main.py  # 或者根据你的环境使用 python main.py
