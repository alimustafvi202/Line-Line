@echo off
title CP2102 Driver Auto-Installer (ESP32 ECG)
echo ======================================================================
echo   SILICON LABS CP2102 USB DRIVER INSTALLER FOR ESP32 + AD8232 ECG
echo ======================================================================
echo.

:: Check for administrative rights
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Administrator privileges required. Requesting UAC elevation...
    powershell -NoProfile -Command "Start-Process cmd.exe -ArgumentList '/c \"\"%~f0\"\"' -Verb RunAs"
    exit /b
)

echo [1/2] Installing CP2102 driver into Windows DriverStore...
cd /d "%~dp0drivers\extracted"
pnputil /add-driver slabvcp.inf /install

echo.
echo [2/2] Running setup verification...
start /wait CP210xVCPInstaller_x64.exe /q

echo.
echo ======================================================================
echo [SUCCESS] Driver installation completed!
echo Checking COM port status...
echo ======================================================================
powershell -NoProfile -Command "[System.IO.Ports.SerialPort]::GetPortNames(); Get-PnpDevice -FriendlyName '*CP210*', '*Silicon Labs*'"
echo.
echo You can now close this window and proceed with the ECG web app.
pause
