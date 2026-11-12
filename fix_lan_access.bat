@echo off
setlocal
set PORT=7777

net session >nul 2>&1
if errorlevel 1 (
  echo.
  echo [!] Please RIGHT-CLICK this file and choose "Run as administrator".
  echo.
  pause
  exit /b 1
)

echo [1/4] checking the console on port %PORT% ...
curl.exe -s -o nul -m 3 http://127.0.0.1:%PORT%/api/ping
if errorlevel 1 (
  echo.
  echo [!] Nothing answers on http://127.0.0.1:%PORT%/
  echo     Start the console first: open Ubuntu, then run
  echo         ./run_server.sh %PORT%
  echo.
  pause
  exit /b 1
)
echo       OK - the console is running.

echo [2/4] opening port %PORT% in Windows Firewall ...
netsh advfirewall firewall add rule name="beatfcn-console-%PORT%" dir=in action=allow protocol=TCP localport=%PORT% >nul
echo       done.

echo [3/4] forwarding port %PORT% from Windows to WSL ...
netsh interface portproxy delete v4tov4 listenport=%PORT% listenaddress=0.0.0.0 >nul 2>&1
netsh interface portproxy add v4tov4 listenport=%PORT% listenaddress=0.0.0.0 connectaddress=127.0.0.1 connectport=%PORT% >nul
echo       done.

echo [4/4] your phone should open one of these:
for /f "tokens=2 delims=:" %%A in ('ipconfig ^| findstr /C:"IPv4"') do call :show "%%A"
echo.
echo Keep the phone on the SAME Wi-Fi as this PC. If still nothing, send this window to the assistant.
echo.
pause
exit /b 0

:show
set "IP=%~1"
set "IP=%IP: =%"
echo       http://%IP%:%PORT%
exit /b 0
