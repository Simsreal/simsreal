#!/bin/bash
python ./config/WSL_ip.py &
/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe "python .\config\Windows_ip.py" &
wait
