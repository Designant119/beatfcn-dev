@echo off
chcp 65001 >nul
setlocal

rem wsl.exe 默认输出 UTF-16，findstr 匹配不到会误判"没装过"；要求 wsl 输出 UTF-8（WSL 0.64+）
set "WSL_UTF8=1"

set "NAME=beatfcn"
set "PORT=7777"
set "TAR=%~d0\beatfcn-train.tar"
set "DST=%~d0\wsl\beatfcn"

echo ============================
echo   BEATFCN 训练系统 一键安装
echo   双击本文件就行，不用敲命令
echo ============================

rem ---- 第 1 步：WSL 功能开没开 ----
wsl --status >nul 2>&1
if errorlevel 1 (
    echo [1/3] WSL 还没开。下面会弹一个管理员窗口，点"是"。
    echo       装完电脑要重启一次，重启回来再双击本文件。
    powershell -NoProfile -Command "Start-Process cmd -Verb RunAs -ArgumentList '/c wsl --install --no-distribution & pause'"
    pause
    goto :eof
)
echo [1/3] WSL 已开

rem ---- 第 2 步：系统导没导进来 ----
set "IMPORTED=1"
wsl -l -q 2>nul | findstr /i "%NAME%" >nul || set "IMPORTED="
if not defined IMPORTED if exist "%DST%\ext4.vhdx" set "IMPORTED=1"
if not defined IMPORTED (
    if not exist "%TAR%" (
        echo [2/3] 没找到 %TAR%
        echo       训练盘没插，或者 tar 不在盘根目录。插好再双击。
        pause
        goto :eof
    )
    echo [2/3] 正在导入，几分钟，别关这个窗口...
    wsl --import %NAME% "%DST%" "%TAR%"
    if errorlevel 1 (
        echo 导入失败。把窗口截图发我。
        pause
        goto :eof
    )
) else (
    echo [2/3] 系统已在，跳过
)

rem ---- 第 3 步：起后端，等端口通了再开网页 ----
echo [3/3] 启动操作台...
wsl -d %NAME% -e bash -lc "cd $(ls -d /home/*/beatfcn-dev /root/beatfcn-dev 2>/dev/null | head -n1) && if command -v tmux >/dev/null 2>&1; then tmux has-session -t console 2>/dev/null || tmux new -ds console 'python3 console_server.py 7777'; else (nohup python3 console_server.py 7777 >/tmp/console.log 2>&1 &); fi"

echo 等后端起来...
powershell -NoProfile -Command "$p=7777; for($i=0;$i -lt 20;$i++){ if((Test-NetConnection -ComputerName 127.0.0.1 -Port $p -WarningAction SilentlyContinue).TcpTestSucceeded){ exit 0 }; Start-Sleep -Milliseconds 800 }; exit 1"
if errorlevel 1 (
    echo 后端没在 7777 上起来。
    echo 诊断：wsl -d %NAME% tail -n 30 /tmp/console.log
    pause
    goto :eof
)

start http://localhost:%PORT%
echo.
echo 好了。浏览器没弹就手动打开 http://localhost:7777
echo 以后想看状态：再双击本文件，已装好的会直接跳到这一步。
pause
