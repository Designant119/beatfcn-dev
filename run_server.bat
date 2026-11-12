@echo off
rem Start the training console from Windows: runs run_server.sh inside WSL.
rem Change DISTRO if you imported it under another name (check with: wsl -l -v)
rem Change the repo path below if it is not at ~/beatfcn-dev inside WSL.
set DISTRO=beatfcn
wsl.exe -d %DISTRO% bash -lc "cd ~/beatfcn-dev && bash run_server.sh 7777"
pause
