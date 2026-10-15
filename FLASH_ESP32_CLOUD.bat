@echo off
title Flash ESP32 Cloud Telemedicine Firmware (Karachi to Haripur)
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\auto_flash_cloud.ps1"
pause
