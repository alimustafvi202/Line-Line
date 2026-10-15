$cli = "d:\Website\ECG-Heart-Anomaly-Research\tools\arduino-cli\arduino-cli.exe"
$sketch = "d:\Website\ECG-Heart-Anomaly-Research\firmware\esp32_ad8232"
$fqbn = "esp32:esp32:esp32:UploadSpeed=115200"
$port = "COM4"

Write-Host "=========================================================="
Write-Host "       CARDIO-AI: AUTOMATED ESP32 FIRMWARE FLASHER        "
Write-Host "=========================================================="
Write-Host "1. Compiling 250 Hz AD8232 ECG firmware..."
& $cli compile --fqbn $fqbn $sketch

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[ERROR] Compilation failed. Please check core installation."
    exit 1
}

Write-Host "`n2. Uploading firmware to $port..."
& $cli upload -p $port --fqbn $fqbn $sketch

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[SUCCESS] Firmware uploaded successfully!"
    Write-Host "ESP32 is now streaming clean 250 Hz ECG data on $port."
} else {
    Write-Host "`n[NOTICE] If upload timed out, hold down the 'BOOT' button on your ESP32 board and run this script again."
}
