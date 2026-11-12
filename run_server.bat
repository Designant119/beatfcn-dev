@echo off
rem Start the training console from Windows: it runs run_server.sh inside WSL.
rem If the repo is not at ~/beatfcn-dev in WSL, change the path below.
wsl.exe bash -lc "cd ~/beatfcn-dev && bash run_server.sh 7777"
pause
