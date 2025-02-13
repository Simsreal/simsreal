#!/bin/bash
python /mnt/e/simsreal/simsreal/config/WSL_ip.py &
/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe "python E:\simsreal\simsreal\config\Windows_ip.py" &
wait
