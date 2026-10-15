@echo off
title Auto-Flash ESP32 AD8232 ECG Firmware
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\auto_flash.ps1"
pause
