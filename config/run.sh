#!/bin/bash
python ./config/WSL_ip.py &
POWERSHELL_PATH=$(which pwsh || which powershell.exe)
if [ -z "$POWERSHELL_PATH" ]; then
    echo "PowerShell not found"
    exit 1
fi
"$POWERSHELL_PATH" "python .\config\Windows_ip.py" &
wait
